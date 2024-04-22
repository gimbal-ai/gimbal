from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2_1
from mediapipe.framework.deps import proto_descriptor_pb2 as _proto_descriptor_pb2
from mediapipe.framework.tool import calculator_graph_template_argument_pb2 as _calculator_graph_template_argument_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CalculatorGraphTemplate(_message.Message):
    __slots__ = ["config", "rule"]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    RULE_FIELD_NUMBER: _ClassVar[int]
    config: _calculator_pb2.CalculatorGraphConfig
    rule: _containers.RepeatedCompositeFieldContainer[TemplateExpression]
    def __init__(self, config: _Optional[_Union[_calculator_pb2.CalculatorGraphConfig, _Mapping]] = ..., rule: _Optional[_Iterable[_Union[TemplateExpression, _Mapping]]] = ...) -> None: ...

class TemplateExpression(_message.Message):
    __slots__ = ["arg", "field_type", "field_value", "key_type", "op", "param", "path"]
    ARG_FIELD_NUMBER: _ClassVar[int]
    FIELD_TYPE_FIELD_NUMBER: _ClassVar[int]
    FIELD_VALUE_FIELD_NUMBER: _ClassVar[int]
    KEY_TYPE_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    PARAM_FIELD_NUMBER: _ClassVar[int]
    PATH_FIELD_NUMBER: _ClassVar[int]
    arg: _containers.RepeatedCompositeFieldContainer[TemplateExpression]
    field_type: _proto_descriptor_pb2.FieldDescriptorProto.Type
    field_value: str
    key_type: _containers.RepeatedScalarFieldContainer[_proto_descriptor_pb2.FieldDescriptorProto.Type]
    op: str
    param: str
    path: str
    def __init__(self, param: _Optional[str] = ..., op: _Optional[str] = ..., arg: _Optional[_Iterable[_Union[TemplateExpression, _Mapping]]] = ..., path: _Optional[str] = ..., field_type: _Optional[_Union[_proto_descriptor_pb2.FieldDescriptorProto.Type, str]] = ..., key_type: _Optional[_Iterable[_Union[_proto_descriptor_pb2.FieldDescriptorProto.Type, str]]] = ..., field_value: _Optional[str] = ...) -> None: ...

class TemplateSubgraphOptions(_message.Message):
    __slots__ = ["dict"]
    DICT_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    dict: _calculator_graph_template_argument_pb2.TemplateDict
    ext: _descriptor.FieldDescriptor
    def __init__(self, dict: _Optional[_Union[_calculator_graph_template_argument_pb2.TemplateDict, _Mapping]] = ...) -> None: ...
