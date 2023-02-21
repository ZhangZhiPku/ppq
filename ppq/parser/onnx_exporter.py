import json
from typing import Union

import numpy as np
import onnx
import torch
from onnx import helper, numpy_helper
from ppq.core import (GRAPH_OPSET_ATTRIB, ONNX_EXPORT_OPSET, ONNX_VERSION,
                      PPQ_CONFIG, DataType, QuantizationStates,
                      convert_any_to_numpy, ppq_warning)
from ppq.IR import (BaseGraph, GraphExporter, Operation, OperationExporter,
                    Variable)
from ppq.IR.quantize import QuantableOperation


class ConstantOfShapeExporter(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # PATCH 20211203, ConstantOfShape Op causes an export error.
        # 这一问题是由 ConstantOfShape 中的 value 格式问题引发的，下面的代码将导出正确的格式
        op.attributes['value'] = numpy_helper.from_array(op.attributes['value'])
        return op

class MMCVExporter(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # MMCV operation must have a domain attribute.
        op.attributes['domain'] = 'mmcv'
        return op

class InterpExporter(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # PATCH 20211216, interp op can not export input_shape attribute.
        op.attributes.pop('input_shape')
        return op

class OOSExporter(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # MMCV operation must have a domain attribute.
        op.attributes['domain'] = 'com.microsoft'
        return op

class AttentionExporter(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        # MMCV operation must have a domain attribute.
        op.attributes['domain'] = 'com.microsoft'
        return op

class PPQBiasFusedMatMulExporter(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.num_of_input == 3: bias = op.inputs[-1]
        assert bias.is_parameter and bias.value is not None, 'MatMul Format Error'
 
        bias_op = graph.create_operation(op_type='Add')
        op.type = 'MatMul'
        graph.insert_op_after(bias_op, op)
        graph.create_variable(value=bias.value, is_parameter=True, dest_ops=[bias_op])
        graph.remove_variable(op.inputs[-1])

class PPQBiasFusedMatMulExporter(OperationExporter):
    def export(self, op: Operation, graph: BaseGraph, **kwargs) -> Operation:
        if op.num_of_input == 3: bias = op.inputs[-1]
        assert bias.is_parameter and bias.value is not None, 'MatMul Format Error'
 
        bias_op = graph.create_operation(op_type='Add')
        op.type = 'MatMul'
        graph.insert_op_after(bias_op, op)
        graph.create_variable(value=bias.value, is_parameter=True, dest_ops=[bias_op])
        graph.remove_variable(op.inputs[-1])

OP_CONVERTERS = {
    'ConstantOfShape': ConstantOfShapeExporter,
    'MMCVRoiAlign': MMCVExporter,
    'grid_sampler': MMCVExporter,
    'Interp': InterpExporter,
    'Attention': AttentionExporter,
    'QAttention': OOSExporter,
    'QGemm': OOSExporter,
    'QLinearAdd': OOSExporter,
    'QLinearAveragePool': OOSExporter,
    'QLinearConcat': OOSExporter,
    'QLinearConv': OOSExporter,
    'QLinearGlobalAveragePool': OOSExporter,
    'QLinearLeakyRelu': OOSExporter,
    'QLinearMul': OOSExporter,
    'QLinearReduceMean': OOSExporter,
    'QLinearSigmoid': OOSExporter,
    'PPQBiasFusedMatMul': PPQBiasFusedMatMulExporter
}

def convert_value(value: Union[int, float, np.ndarray, torch.Tensor]) -> str:
    if type(value) in {int, float}: return value
    else:
        value = convert_any_to_numpy(value, accept_none=True)
        if value is None: return value # SOI config has Nona as its scale and
        return value.tolist()


class OnnxExporter(GraphExporter):
    """
    PPQ 可以将 计算图 导出成 Onnx 标准格式，Onnx Exporter 不会导出 QDQ 节点。
    如需导出带有 QDQ 节点的 Onnx 文件，用户需要使用 OnnxRuntime Exporter
    
    任何导出器的导出逻辑都是原地进行的，它们将对传入的计算图对象进行原地修改，因此在导出之前你需要手动克隆计算图。
    """
    def __init__(self) -> None:
        super().__init__()

    def export_quantization_config(self, config_path: str, graph: BaseGraph):
        """Export Tensor Quantization Config to File(Json)."""

        render_buffer = {
            'configs': {},
            'dispatchings' : {},
            'values': {}
        }

        # Render quantization config.
        for operation in graph.operations.values():
            if isinstance(operation, QuantableOperation):
                op_dict = {
                    var.name: {
                        'bit_width'  : config.num_of_bits,
                        'policy'     : config.policy.to_dict(),
                        'state'      : config.state.name,
                        'quant_min'  : config.quant_min,
                        'quant_max'  : config.quant_max,
                        'hash'       : config.__hash__(),
                        'dominator'  : config.dominated_by.__hash__()
                    }
                    for config, var in operation.config_with_variable
                }

                for config, _ in operation.config_with_variable:
                    if config.dominated_by == config:
                        if (config.state != QuantizationStates.FP32):
                            render_buffer['values'][config.__hash__()] = {
                                'scale'      : convert_value(config.scale),
                                'zero_point' : convert_value(config.offset),
                            }

                render_buffer['configs'][operation.name] = op_dict
                render_buffer['dispatchings'][operation.name] = operation.platform.name

        with open(file=config_path, mode='w') as file:
            json.dump(render_buffer, file, indent=4)

    def export_graph(self, graph: BaseGraph) -> onnx.ModelProto:
        """
        Convert a PPQ IR to Onnx IR.
        This export will only convert PPQ Op and var to onnx, all quantization configs will be skipped.
        
        This function will try to keep the opset version of your graph unchanged. 
        However if the opset is not given, ppq will convert it to with the global parameter ppq.core.ONNX_EXPORT_OPSET.
        """
        name = graph._name
        if not name: name = f'{PPQ_CONFIG.NAME} - v({PPQ_CONFIG.VERSION})'

        # Ready to export onnx graph defination.
        _inputs, _outputs, _initilizers, _nodes, _value_info = [], [], [], [], []
        
        # before we can export them, we firstly convert all ops to proper format.
        for op in [_ for _ in graph.topological_sort()]:
            if op.type in OP_CONVERTERS:
                exporter = OP_CONVERTERS[op.type]()
                assert isinstance(exporter, OperationExporter), (
                    f'Expected an OpExporter here, however {type(exporter)} was given.')
                op = exporter.export(op=op, graph=graph)
        
        for op in graph.topological_sort():
            _nodes.append(self.build_operator_proto(op))

        for variable in graph.variables.values():
            tensor_proto = self.build_variable_proto(variable)
            if variable.name in graph.inputs:
                _inputs.append(tensor_proto)
            if variable.name in graph.outputs:
                _outputs.append(tensor_proto)
            if variable.is_parameter:
                _initilizers.append(tensor_proto)
            else:
                _value_info.append(tensor_proto)

        graph_def = helper.make_graph(
            name=name, nodes=_nodes,
            inputs=_inputs, outputs=_outputs,
            initializer=_initilizers, 
            value_info=_value_info)

        # if opset is missing from your graph, give it a default one.
        if GRAPH_OPSET_ATTRIB not in graph._detail:
            op = onnx.OperatorSetIdProto()
            op.version = ONNX_EXPORT_OPSET
            opsets = [op]
        else:
            opsets = []
            for opset in graph._detail[GRAPH_OPSET_ATTRIB]:
                op = onnx.OperatorSetIdProto()
                op.domain = opset['domain']
                op.version = opset['version']
                opsets.append(op)

        onnx_model = helper.make_model(
            graph_def, producer_name=PPQ_CONFIG.NAME, opset_imports=opsets)
        onnx_model.ir_version = graph._detail.get('ir_version', ONNX_VERSION)
        return onnx_model

    def build_operator_proto(self, operation: Operation) -> onnx.OperatorProto:
        """
        Convert PPQ Op to Onnx Operation
        An Op consumes zero or more Tensors, and produces zero or more Tensors.
        """
        attributes = operation.attributes
        for key in attributes:
            value = attributes[key]
            if isinstance(value, DataType):
                attributes[key] = value.value
            if isinstance(value, torch.Tensor):
                if value.numel() == 0: attributes[key] = None
                elif value.numel() == 1: attributes[key] = convert_any_to_numpy([value.item()]) # convert to 1d array
                else: attributes[key] = convert_any_to_numpy(value)

        if PPQ_CONFIG.EXPORT_PPQ_INTERNAL_INFO:
            attributes['platform'] = operation.platform.name

        op_proto = helper.make_node(
            op_type=operation.type,
            inputs=[_.name for _ in operation.inputs],
            outputs=[_.name for _ in operation.outputs],
            name=operation.name,
            **attributes)

        return op_proto

    def build_variable_proto(self, variable: Variable) -> onnx.TensorProto:
        """
        Convert PPQ Variable to Onnx TensorProto, There are 2 different types of Tensor in Onnx:
            Variable: Represents a Tensor whose value is not known until inference-time.
            Constant: Represents a Tensor whose value is known.
        """
        # Parameter Varaible in PPQ, Constant Variable in Onnx
        if variable.is_parameter:
            if variable.value is not None:
                var_shape     = variable.value.shape
                pytorch_dtype = variable.value.dtype
                onnx_dtype    = DataType.convert_from_torch(pytorch_dtype).value
 
        # Non Parameter
        else:
            var_shape  = variable.shape
            onnx_dtype = variable.dtype.value

        if not variable.is_parameter:
            tensor_proto = helper.make_tensor_value_info(
                name=variable.name, elem_type=onnx_dtype, shape=var_shape)
        else:
            value = variable.value
            is_raw_format = False
            if isinstance(value, torch.Tensor):
                if value.numel() == 0: value = []
                elif value.ndim >= 1:
                    value = convert_any_to_numpy(variable.value).flatten()
                    value = value.tobytes()
                    is_raw_format = True
                elif value.ndim == 0: # Pytorch Scalar Type
                    value = [value.item(), ] # it is fine for onnx, shape for this value will be []
            else: value = value # value is python primary type.
            tensor_proto = helper.make_tensor(
                name=variable.name, data_type=onnx_dtype,
                dims=var_shape, vals=value, raw=is_raw_format)
        return tensor_proto

    def export(self, file_path: str, graph: BaseGraph, 
               config_path: str = None, save_as_external_data: bool=False):
        # during export we will remove all boundary operations from graph.
        # we do not want to change the structure of original graph,
        # so there have to take a clone of it.
        # graph = graph.copy()

        # if a valid config path is given, export quantization config to there.
        if config_path is not None:
            self.export_quantization_config(config_path, graph)

        size_threshold = 0 if save_as_external_data else 1024
        onnx.save(self.export_graph(graph=graph), file_path, 
                  size_threshold=size_threshold,
                  save_as_external_data=save_as_external_data,
                  all_tensors_to_one_file=(not save_as_external_data))
