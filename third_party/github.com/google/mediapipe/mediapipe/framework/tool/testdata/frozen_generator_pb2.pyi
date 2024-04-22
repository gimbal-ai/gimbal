from mediapipe.framework import packet_generator_pb2 as _packet_generator_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FrozenGeneratorOptions(_message.Message):
    __slots__ = ["graph_proto_path", "initialization_op_names", "tag_to_tensor_names"]
    class TagToTensorNamesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    EXT_FIELD_NUMBER: _ClassVar[int]
    GRAPH_PROTO_PATH_FIELD_NUMBER: _ClassVar[int]
    INITIALIZATION_OP_NAMES_FIELD_NUMBER: _ClassVar[int]
    TAG_TO_TENSOR_NAMES_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    graph_proto_path: str
    initialization_op_names: _containers.RepeatedScalarFieldContainer[str]
    tag_to_tensor_names: _containers.ScalarMap[str, str]
    def __init__(self, graph_proto_path: _Optional[str] = ..., tag_to_tensor_names: _Optional[_Mapping[str, str]] = ..., initialization_op_names: _Optional[_Iterable[str]] = ...) -> None: ...
