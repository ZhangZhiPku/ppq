from typing import List, Tuple

import numpy as np
import onnx
import torch
from onnx import helper
from enum import Enum
from ppq.core import (EXPORT_DEVICE_SWITCHER, ORT_MICROSOFT_CONTRIB_LINEAR_OPS,
                      ORT_OOS_FUSE_START_OPS, PPQ_NAME, DataType,
                      OperationMeta, QuantizationProperty, QuantizationStates,
                      TensorMeta, TensorQuantizationConfig,
                      convert_any_to_torch_tensor)
from ppq.IR import BaseGraph, Operation, QuantableVariable, Variable
from ppq.IR.morph import GraphDeviceSwitcher
from ppq.IR.quantize import QuantableOperation
from ppq.core.common import PASSIVE_OPERATIONS
from ppq.core.defs import ppq_warning
from ppq.quantization.qfunction.linear import PPQLinearQuant_toInt
from ppq.utils.round import ppq_tensor_round

from .onnxruntime_exporter import ONNXRUNTIMExporter


class QuantJointOpInputOrder(Enum):
    INPUT_FIRST = 1
    OUTPUT_FIRST = 2


class ORTOOSExporter(ONNXRUNTIMExporter):
    ASYMMETRICAL_ZP_NP_TYPE = torch.uint8
    SYMMETRICAL_ZP_NP_TYPE = torch.int8
    BIAS_NP_TYPE = torch.int32
    SCALE_NP_TYPE = torch.float32

    SCALE_PARAMETER_SUFFIX = "_scale"
    ZP_PARAMETER_SUFFIX = "_zero_point"
    QUANTIZE_PARAMETER_SUFFIX = "_quantized"
    WEIGHT_QUANTIZE_PARAMETER_SUFFIX = "_weight"
    BIAS_QUANTIZE_PARAMETER_SUFFIX = "_bias"
    LINKER_VAR_SUFFIX = "_linker"
    QUANTIZE_LINEAR_SUFFIX = "_QuantizeLinear"
    DEQUANTIZE_LINEAR_SUFFIX = "_DequantizeLinear"

    @property
    def qlinear_op_map(self):
        return {
            "Add": "QLinearAdd",
            "Mul": "QLinearMul",
            "AveragePool": "QLinearAveragePool",
            "Conv": "QLinearConv",
            "GlobalAveragePool": "QLinearGlobalAveragePool",
            "MatMul": "QLinearMatMul",
            "Concat": "QLinearConcat",
        }

    @property
    def qlinear_ops(self):
        return self.qlinear_op_map.values()

    def get_qlinear_op_type(self, op_type: str) -> str:
        return self.qlinear_op_map.get(op_type, op_type)

    def get_qlinear_op_dominant_dtype(self, op: Operation) -> np.dtype:
        if op.type in [
            "QLinearConv",
            "QLinearAveragePool",
            "QLinearGlobalAveragePool",
            "QLinearMatMul",
            "QLinearAdd",
            "QLinearMul",
            "QLinearConcat",
        ]:
            # align with zp dtype
            return op.inputs[2].meta.dtype
        raise NotImplementedError(
            f"Please implement dominant dtype extraction for {op.type}"
        )

    @classmethod
    def get_dtype_on_symmetricity(cls, is_asymmetrical: bool) -> torch.dtype:
        return (
            cls.ASYMMETRICAL_ZP_NP_TYPE
            if is_asymmetrical
            else cls.SYMMETRICAL_ZP_NP_TYPE
        )

    @classmethod
    def is_quantize_parameter_added(cls, var: Variable, graph: BaseGraph) -> bool:
        return var.name + cls.SCALE_PARAMETER_SUFFIX in graph.variables

    def is_quantized_qlinear_op(self, op: Operation) -> bool:
        return (
            op is not None
            and op.type in self.qlinear_op_map
            and isinstance(op, QuantableOperation)
        )

    def is_asymmetrical(self, config: TensorQuantizationConfig) -> bool:
        return config.policy.has_property(QuantizationProperty.ASYMMETRICAL)

    def is_per_channel(self, config: TensorQuantizationConfig) -> bool:
        return config.policy.has_property(QuantizationProperty.PER_CHANNEL)

    def get_canonized_quantization_parameters(
        self, var: Variable, quantize_config: TensorQuantizationConfig
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # So far canonization only reduces dimensionality 1 -> 0,
        # which is just a workaround for `QuantAlignmentPass`. See method: `align_to_large`
        # For OOS style ONNX quantized model, scale & zp with shape like [0.xxx]
        # are not accepted for activations
        scale, offset = quantize_config.scale, quantize_config.offset
        if var.is_parameter is False:
            if len(scale.shape) != 0:
                scale = torch.tensor(scale.item(), dtype=scale.dtype)
            if len(offset.shape) != 0:
                offset = torch.tensor(offset.item(), dtype=offset.dtype)
        return scale, offset

    def build_per_channel_param_broadcast_shape(
        self, weight: torch.Tensor, param: torch.Tensor
    ) -> torch.Tensor:
        prefix_count = 0
        suffix_count = 0
        while weight.shape[prefix_count] != param.shape[0]:
            prefix_count += 1
        suffix_count = len(weight.shape) - prefix_count - 1
        return param[(None,) * prefix_count + (...,) + (None,) * suffix_count]

    def quantize_weight(
        self,
        weight: torch.Tensor,
        scale: torch.Tensor,
        zero_point: torch.Tensor,
        is_asymmetrical: bool,
        is_per_channel: bool,
    ) -> torch.Tensor:
        weight_dtype = self.get_dtype_on_symmetricity(is_asymmetrical)
        if is_per_channel is True:
            unsqueezed_scale = self.build_per_channel_param_broadcast_shape(
                weight, scale
            )
            unsqueezed_zp = self.build_per_channel_param_broadcast_shape(
                weight, zero_point
            )
            return (
                ((weight / unsqueezed_scale).round() + unsqueezed_zp)
                .cpu()
                .to(weight_dtype)
            )
        else:
            return ((weight / scale).round() + zero_point).cpu().to(weight_dtype)

    def quantize_bias(
        self, bias: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
    ) -> torch.Tensor:
        return (
            ((bias / scale).round() + zero_point).cpu().to(ORTOOSExporter.BIAS_NP_TYPE)
        )

    def add_scale_and_zp_parameter(
        self,
        var: Variable,
        graph: BaseGraph,
        dest_index: int,
        quantize_config: TensorQuantizationConfig,
        link_to_source=False,
    ) -> Tuple[Variable]:
        if self.is_quantize_parameter_added(var, graph):
            scale = graph.variables[var.name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX]
            offset = graph.variables[var.name + ORTOOSExporter.ZP_PARAMETER_SUFFIX]
            scale.dest_ops.append(var.dest_ops[dest_index])
            offset.dest_ops.append(var.dest_ops[dest_index])
            if link_to_source:
                scale.dest_ops.append(var.source_op)
                offset.dest_ops.append(var.source_op)
        else:
            is_asymmetrical = self.is_asymmetrical(quantize_config)
            offset_dtype = self.get_dtype_on_symmetricity(is_asymmetrical)
            scale, offset = self.get_canonized_quantization_parameters(
                var, quantize_config
            )
            dest_ops = (
                [var.source_op, var.dest_ops[dest_index]]
                if link_to_source
                else [var.dest_ops[dest_index]]
            )
            scale = Variable(
                name=var.name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX,
                value=convert_any_to_torch_tensor(
                    scale, dtype=ORTOOSExporter.SCALE_NP_TYPE
                ),
                is_parameter=True,
                dest_ops=dest_ops,
            )
            offset = Variable(
                name=var.name + ORTOOSExporter.ZP_PARAMETER_SUFFIX,
                value=convert_any_to_torch_tensor(offset, dtype=offset_dtype),
                is_parameter=True,
                dest_ops=dest_ops,
            )
            graph.append_variable(scale)
            graph.append_variable(offset)
        return scale, offset

    def add_quantized_parameter(
        self,
        var: Variable,
        graph: BaseGraph,
        dest_index: int,
        quantize_config: TensorQuantizationConfig,
        quantize_suffix: str,
        is_bias: bool,
    ) -> Variable:
        scale, offset = quantize_config.scale, quantize_config.offset
        if is_bias:
            quant_value = self.quantize_bias(var.value, scale, offset)
        else:
            is_asymmetrical = self.is_asymmetrical(quantize_config)
            is_per_channel = self.is_per_channel(quantize_config)
            quant_value = self.quantize_weight(
                var.value, scale, offset, is_asymmetrical, is_per_channel
            )
        # quant_val = graph.create_variable(value=quant_value, is_parameter=True) 
        Variable(
            name=var.name + quantize_suffix,
            value=quant_value,
            is_parameter=True,
            dest_ops=[var.dest_ops[dest_index]],
        )
        graph.append_variable(quant_val)
        return quant_val

    def add_quantize_linear_op_quant_parameter(
        self, graph: BaseGraph, var: Variable, index: int
    ) -> None:
        if self.is_quantize_parameter_added(var, graph):
            # quantization parameter would be shared by multiple operations
            graph.variables[
                var.name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX
            ].dest_ops.append(var.dest_ops[index])
            graph.variables[
                var.name + ORTOOSExporter.ZP_PARAMETER_SUFFIX
            ].dest_ops.append(var.dest_ops[index])
        else:
            # add new quantization parameter
            quantize_config = var.source_op_config
            self.add_scale_and_zp_parameter(
                var, graph, index, quantize_config, link_to_source=True
            )

    def add_quant_parameter_for_op(
        self, graph: BaseGraph, var: Variable
    ) -> Tuple[Variable]:
        assert len(var.dest_ops) == 1
        quantize_config = var.dest_op_configs[0]
        scale_val, offset_val = self.add_scale_and_zp_parameter(
            var, graph, 0, quantize_config, link_to_source=False
        )
        quant_val = self.add_quantized_parameter(
            var,
            graph,
            0,
            quantize_config,
            ORTOOSExporter.QUANTIZE_PARAMETER_SUFFIX,
            is_bias=False,
        )
        return quant_val, scale_val, offset_val

    def add_quant_parameter_for_conv_op(
        self, graph: BaseGraph, var: Variable, is_bias: bool
    ) -> Tuple[Variable]:
        assert len(var.dest_ops) == 1
        quantize_config = var.dest_op_configs[0]
        if is_bias is True:
            quant_val = self.add_quantized_parameter(
                var,
                graph,
                0,
                quantize_config,
                ORTOOSExporter.BIAS_QUANTIZE_PARAMETER_SUFFIX
                + ORTOOSExporter.QUANTIZE_PARAMETER_SUFFIX,
                is_bias=True,
            )
        else:
            scale_val, offset_val = self.add_scale_and_zp_parameter(
                var, graph, 0, quantize_config, link_to_source=False
            )
            quant_val = self.add_quantized_parameter(
                var,
                graph,
                0,
                quantize_config,
                ORTOOSExporter.WEIGHT_QUANTIZE_PARAMETER_SUFFIX
                + ORTOOSExporter.QUANTIZE_PARAMETER_SUFFIX,
                is_bias=False,
            )
        if is_bias is False:
            return quant_val, scale_val, offset_val
        return quant_val

    def insert_quant_Linear_operation(
        self, graph: BaseGraph, var: Variable, index: int
    ) -> Operation:
        quantize_config = var.dest_op_configs[index]
        quant_op = Operation(
            name=var.name + ORTOOSExporter.QUANTIZE_LINEAR_SUFFIX,
            op_type="QuantizeLinear",
            attributes={},
        )
        graph.append_operation(quant_op)
        link_var = Variable(
            name=var.name + ORTOOSExporter.LINKER_VAR_SUFFIX,
            dest_ops=[],
            source_op=quant_op,
        )
        graph.append_variable(link_var)
        quant_op.inputs.append(var)
        quant_op.outputs.append(link_var)
        var.dest_ops[index] = quant_op
        scale_val, offset_val = self.add_scale_and_zp_parameter(
            var, graph, index, quantize_config, link_to_source=False
        )
        quant_op.inputs.extend([scale_val, offset_val])
        return quant_op

    def insert_dequant_Linear_operation(
        self, graph: BaseGraph, var: Variable, index: int, is_output_var=False
    ) -> Operation:
        quantize_config = var.source_op_config
        dequant_op = Operation(
            name=var.name + ORTOOSExporter.DEQUANTIZE_LINEAR_SUFFIX,
            op_type="DequantizeLinear",
            attributes={},
        )
        graph.append_operation(dequant_op)
        link_var = Variable(
            name=var.name + ORTOOSExporter.LINKER_VAR_SUFFIX,
            dest_ops=[],
            source_op=dequant_op,
        )
        graph.append_variable(link_var)
        dequant_op.inputs.append(var)
        dequant_op.outputs.append(link_var)
        # case when DequantizeLinear op required to be inserted before output
        if is_output_var:
            var.dest_ops.append(dequant_op)
        else:
            var.dest_ops[index] = dequant_op
        scale_val, offset_val = self.add_scale_and_zp_parameter(
            var,
            graph,
            len(var.dest_ops) - 1 if is_output_var else index,
            quantize_config,
            link_to_source=False,
        )
        dequant_op.inputs.extend([scale_val, offset_val])
        if is_output_var:
            del graph.outputs[var.name]
            graph.outputs[link_var.name] = link_var
        return dequant_op

    def transform_qlinear_conv(self, graph: BaseGraph, op: Operation) -> None:
        # Input scale
        input_val_name = op.inputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        input_scale, input_offset = (
            graph.variables[input_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[input_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        # Weight scale
        weight_val = op.inputs[1]
        (
            weight_quant,
            weight_scale,
            weight_offset,
        ) = self.add_quant_parameter_for_conv_op(graph, weight_val, False)
        # Output scale
        assert len(op.outputs) == 1
        output_val_name = op.outputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        output_scale, output_offset = (
            graph.variables[output_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[output_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        qlinear_conv_inputs = [
            op.inputs[0],
            input_scale,
            input_offset,
            weight_quant,
            weight_scale,
            weight_offset,
            output_scale,
            output_offset,
        ]
        graph.delete_variable(op.inputs[1].name, True)
        # Bias
        if len(op.inputs) == 3:
            bias = op.inputs[2]
            bias_val = self.add_quant_parameter_for_conv_op(graph, bias, True)
            qlinear_conv_inputs.append(bias_val)
            graph.delete_variable(op.inputs[2].name, True)
        op.type = self.qlinear_op_map["Conv"]
        op.inputs.clear()
        op.inputs.extend(qlinear_conv_inputs)

    def transform_qlinear_joint_op(
        self, graph: BaseGraph, op: Operation, input_order: QuantJointOpInputOrder
    ) -> None:
        # Output scale
        assert len(op.outputs) == 1
        output_val_name = op.outputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        output_scale, output_offset = (
            graph.variables[output_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[output_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        qlinear_inputs_for_origin_outputs = [output_scale, output_offset]
        # Input scales
        qlinear_inputs_for_origin_inputs = []
        for index, input_value in enumerate(op.inputs):
            input_val_name = input_value.name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
            if input_value.is_parameter:
                (
                    input_value,
                    input_scale,
                    input_offset,
                ) = self.add_quant_parameter_for_op(graph, input_value)
                graph.delete_variable(op.inputs[index].name, True)
            else:
                input_scale, input_offset = (
                    graph.variables[
                        input_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX
                    ],
                    graph.variables[
                        input_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX
                    ],
                )
            qlinear_inputs_for_origin_inputs.extend(
                [input_value, input_scale, input_offset]
            )
        if op.type in ORT_MICROSOFT_CONTRIB_LINEAR_OPS:
            op.attributes["domain"] = "com.microsoft"
        op.type = self.qlinear_op_map[op.type]
        op.inputs.clear()
        op.inputs.extend(
            qlinear_inputs_for_origin_inputs + qlinear_inputs_for_origin_outputs
            if input_order == QuantJointOpInputOrder.INPUT_FIRST
            else qlinear_inputs_for_origin_outputs + qlinear_inputs_for_origin_inputs
        )

    def transform_qlinear_average_pool(
        self, graph: BaseGraph, op: Operation, is_global=False
    ) -> None:
        # Input scale
        input_val_name = op.inputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        input_scale, input_offset = (
            graph.variables[input_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[input_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        # Output scale
        assert len(op.outputs) == 1
        output_val_name = op.outputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        output_scale, output_offset = (
            graph.variables[output_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[output_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        qlinear_inputs = [
            op.inputs[0],
            input_scale,
            input_offset,
            output_scale,
            output_offset,
        ]
        op.type = self.qlinear_op_map[
            "GlobalAveragePool" if is_global is True else "AveragePool"
        ]
        op.attributes["domain"] = "com.microsoft"
        op.inputs.clear()
        op.inputs.extend(qlinear_inputs)

    def transform_qlinear_matmul(self, graph: BaseGraph, op: Operation) -> None:
        # Input scale 0
        input_val_name_0 = op.inputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        if op.inputs[0].is_parameter:
            input_0, input_scale_0, input_offset_0 = self.add_quant_parameter_for_op(
                graph, op.inputs[0]
            )
            graph.delete_variable(op.inputs[0].name, True)
        else:
            input_0 = op.inputs[0]
            input_scale_0, input_offset_0 = (
                graph.variables[
                    input_val_name_0 + ORTOOSExporter.SCALE_PARAMETER_SUFFIX
                ],
                graph.variables[input_val_name_0 + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
            )
        # Input scale 1
        input_val_name_1 = op.inputs[1].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        if op.inputs[1].is_parameter:
            input_1, input_scale_1, input_offset_1 = self.add_quant_parameter_for_op(
                graph, op.inputs[1]
            )
            graph.delete_variable(op.inputs[1].name, True)
        else:
            input_1 = op.inputs[1]
            input_scale_1, input_offset_1 = (
                graph.variables[
                    input_val_name_1 + ORTOOSExporter.SCALE_PARAMETER_SUFFIX
                ],
                graph.variables[input_val_name_1 + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
            )
        # Output scale
        assert len(op.outputs) == 1
        output_val_name = op.outputs[0].name.split(ORTOOSExporter.LINKER_VAR_SUFFIX)[0]
        output_scale, output_offset = (
            graph.variables[output_val_name + ORTOOSExporter.SCALE_PARAMETER_SUFFIX],
            graph.variables[output_val_name + ORTOOSExporter.ZP_PARAMETER_SUFFIX],
        )
        qlinear_inputs = [
            input_0,
            input_scale_0,
            input_offset_0,
            input_1,
            input_scale_1,
            input_offset_1,
            output_scale,
            output_offset,
        ]
        op.type = self.qlinear_op_map["MatMul"]
        op.inputs.clear()
        op.inputs.extend(qlinear_inputs)

    def transform_qlinear_operation(
        self, operation: Operation, graph: BaseGraph
    ) -> None:
        if operation.type == "Conv":
            self.transform_qlinear_conv(graph, operation)
        if operation.type in ["Add", "Mul"]:
            self.transform_qlinear_joint_op(
                graph, operation, QuantJointOpInputOrder.INPUT_FIRST
            )
        if operation.type == "GlobalAveragePool":
            self.transform_qlinear_average_pool(graph, operation, is_global=True)
        if operation.type == "AveragePool":
            self.transform_qlinear_average_pool(graph, operation, is_global=False)
        if operation.type == "MatMul":
            self.transform_qlinear_matmul(graph, operation)
        if operation.type == "Concat":
            self.transform_qlinear_joint_op(
                graph, operation, QuantJointOpInputOrder.OUTPUT_FIRST
            )

    def permute_op_meta_order(self, op: Operation, curr_meta_len: int) -> None:
        if op.type == "QLinearAdd":
            # the second operand's index move from 1 to 3
            op.meta_data.input_metas[3] = op.meta_data.input_metas[1]
        elif op.type == "QLinearConcat":
            # first two holes are for output scale and zp, thus all input scale/zep needs to move
            # two steps forward
            for i in range(curr_meta_len - 1, -1, -1):
                op.meta_data.input_metas[i + 2] = op.meta_data.input_metas[i]

    def correct_param_meta(self, graph: BaseGraph) -> None:
        # handle QLinear ops
        for op in graph.topological_sort():
            if op.type in self.qlinear_ops:
                curr_meta_len = len(op.meta_data.input_metas)
                expected_len = len(op.inputs)
                for _ in range(expected_len - curr_meta_len):
                    op.meta_data.input_metas.append(TensorMeta(DataType.FP32, None))
                self.permute_op_meta_order(op, curr_meta_len)

        # correct parameter meta data
        for var in graph.variables.values():
            if var.is_parameter:
                for op in var.dest_ops:
                    if op.meta_data is None:
                        op.meta_data = OperationMeta(
                            [TensorMeta(DataType.FP32, None, v.name) for v in op.inputs],
                            [TensorMeta(DataType.FP32, None, v.name) for v in op.outputs],
                            op.name,
                            op.type,
                            -1,
                        )

                    if torch.is_tensor(var.value):
                        new_input_meta = TensorMeta.parsing_from_torch_tensor(
                            var.value, var.name
                        )
                    else:
                        new_input_meta = TensorMeta.parsing_from_numpy_ndarray(
                            var.value, var.name
                        )

                    op.meta_data.input_metas[op.inputs.index(var)] = new_input_meta

        # add variable meta info in topo order
        for op in graph.topological_sort():
            if op.type == "QuantizeLinear" and op.inputs[0].source_op is not None:
                input_var = op.inputs[0]
                op.meta_data.input_metas[0] = input_var.meta
                op.meta_data.output_metas[0].shape = input_var.meta.shape
                op.meta_data.output_metas[0].dtype = op.meta_data.input_metas[2].dtype
            # must be input
            elif op.type == "QuantizeLinear" and op.inputs[0].value is None:
                var = op.outputs[0]
                dest_op = var.dest_ops[0]
                dest_idx = var.dest_idx[0]
                meta = dest_op.meta_data.input_metas[dest_idx]
                # meta can't be None itself because we have built TensorMeta
                # for every input when we correct param meta
                while meta.shape is None or meta.dtype is None:
                    var = dest_op.outputs[0]
                    dest_op = var.dest_ops[0]
                    dest_idx = var.dest_idx[0]
                    meta = dest_op.meta_data.input_metas[dest_idx]

                op.meta_data.input_metas[0] = meta
                op.meta_data.output_metas[0].shape = meta.shape
                op.meta_data.output_metas[0].dtype = op.meta_data.input_metas[2].dtype
            elif op.type == "DequantizeLinear":
                input_var = op.inputs[0]
                op.meta_data.input_metas[0] = input_var.meta
                op.meta_data.output_metas[0].shape = input_var.meta.shape
                op.meta_data.output_metas[0].dtype = op.meta_data.input_metas[1].dtype
            elif op.type in self.qlinear_ops:
                for output_meta in op.meta_data.output_metas:
                    output_meta.dtype = self.get_qlinear_op_dominant_dtype(op)

    def conversion_preprocess(self, op: Operation) -> Tuple[List[Variable], List[TensorMeta]]:
        """
        Detach all input variable from given op, prepare for inserting input variable for it.

        Args:
            op (Operation): _description_

        Returns:
            List[Variable]: all detached variable
        """
        inputs = [var for var in op.inputs]
        input_metas = op.meta_data.input_metas.copy()
        for var in op.inputs:
            var.dest_ops.remove(op)
        op.inputs.clear()
        op.meta_data.input_metas.clear()
        return inputs, input_metas

    def convert_operation(self, graph: BaseGraph, op: QuantableOperation, 
                          process_activation: bool, process_parameter: bool, 
                          quant_param_to_int: bool):
        """
        Convert an operation to onnx operator oriented format.
        There are 2 ways to represent quantized ONNX models:

        Operator Oriented. All the quantized operators have their own ONNX definitions,
            like QLinearConv, MatMulInteger and etc.
        
        Tensor Oriented, aka Quantize and DeQuantize (QDQ). 
            This format uses DQ(Q(tensor)) to simulate the quantize and dequantize process, 
            and QuantizeLinear and DeQuantizeLinear operators also carry the quantization parameters. 
            
        Quantization-Aware training (QAT) models converted from Tensorflow or exported from PyTorch.
        
        Quantized models converted from tflite and other framework.

        Args:
            graph (BaseGraph): _description_
            op (QuantableOperation): _description_
            process_activation (bool): _description_
            process_parameter (bool): _description_
            quant_param_to_int (bool): _description_

        Returns:
            _type_: _description_
        """
        if op.type in {
                'Conv', 'Gemm', 'Matmul', 'AveragePool', 'LeakyRelu', 'Sigmoid',
                'GlobalAveragePool', 'Add', 'Mul', 'Concat', 'ReduceMean'}:
            # Those operation can convert to onnx opeartion-oriented quantized op.

            inputs, input_metas = self.conversion_preprocess(op)
            bias, bias_meta, bias_config = None, None, None
            if op.type in {'Conv', 'Gemm'} and len(inputs) > 3: # has bias
                bias        = inputs[-1]
                bias_meta   = bias.meta
                bias_config = op.config.output_quantization_config[-1]
                inputs = inputs[: -1] # remove bias from inputs, process it later.

            # process input
            for config, var, meta in zip(op.config.input_quantization_config, inputs, input_metas):
                otype, vtype = self.infer_qtype(config)
                scale  = convert_any_to_torch_tensor(config.scale.clone(), dtype=torch.float32)
                offset = ppq_tensor_round(config.offset.clone()).type(otype)

                if var.is_parameter:
                    if config.state not in {
                        QuantizationStates.ACTIVATED, QuantizationStates.PASSIVE, 
                        QuantizationStates.PASSIVE_BAKED, QuantizationStates.BAKED}:
                        raise PermissionError(
                            f'Can not export operation {op.name} in onnx operator oriented quantize format, '
                            f'Cause its parameter {var.name} has not been correctly quantized.')

                    if config.num_of_bits != 8:
                        raise PermissionError(
                            f'Can not export operation {op.name} in onnx operator oriented quantize format, '
                            f'Cause its parameter {var.name} is not quantized with 8 bits.')

                    config.state = QuantizationStates.ACTIVATED
                    var.value    = PPQLinearQuant_toInt(tensor=var.value, config=config)

                graph.create_link_with_op(variable=var, upstream_op=None, downstream_op=op)
                graph.create_link_with_op(
                    variable=graph.create_variable(value=scale, is_parameter=True), 
                    upstream_op=None, downstream_op=op)
                graph.create_link_with_op(
                    variable=graph.create_variable(value=offset, is_parameter=True), 
                    upstream_op=None, downstream_op=op)
                op.meta_data.input_metas.extend([
                    TensorMeta(dtype=DataType.convert_from_torch(vtype), shape=meta.shape), 
                    TensorMeta(dtype=DataType.FP32, shape=config.scale.shape), 
                    TensorMeta(dtype=DataType.convert_from_torch(otype), shape=config.offset.shape)])

            # process output
            assert len(op.outputs) == 1, 'Oops seems we got something wrong here.'
            config, var = op.config.output_quantization_config[0], op.outputs[0]
            graph.create_link_with_op(
                variable=graph.create_variable(value=scale, is_parameter=True), 
                upstream_op=None, downstream_op=op)
            graph.create_link_with_op(
                variable=graph.create_variable(value=offset, is_parameter=True), 
                upstream_op=None, downstream_op=op)
            op.meta_data.input_metas.extend([
                TensorMeta(dtype=DataType.FP32, shape=config.scale.shape), 
                TensorMeta(dtype=DataType.convert_from_torch(otype), shape=config.offset.shape)])

            # process bias
            if bias is not None:
                if bias_config: # TODO START HERE
                graph.create_link_with_op(variable=bias, upstream_op=None, downstream_op=op)
                op.meta_data.input_metas.extend([TensorMeta(dtype=DataType.FP32, shape=bias_meta.shape)])

        elif op.type not in PASSIVE_OPERATIONS: # Those operation can not convert to onnx quantized op
            # If opeartion is not a passive operation, skip its conversion is not safe,
            # We have to export it within qdq format.
            ppq_warning(f'Operation {op.name} can not convert to onnx oos quantized format, '
                        'ppq will export it in onnx qdq format.')
            return super().convert_operation(
                graph, op, process_activation, 
                process_parameter, quant_param_to_int)
        else:
            # If operation is passive, skip it is safe.
            pass
