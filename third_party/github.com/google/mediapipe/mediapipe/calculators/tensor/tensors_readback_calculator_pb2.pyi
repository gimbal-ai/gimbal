from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorsReadbackCalculatorOptions(_message.Message):
    __slots__ = ["tensor_shape"]
    class TensorShape(_message.Message):
        __slots__ = ["dims"]
        DIMS_FIELD_NUMBER: _ClassVar[int]
        dims: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, dims: _Optional[_Iterable[int]] = ...) -> None: ...
    EXT_FIELD_NUMBER: _ClassVar[int]
    TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    tensor_shape: _containers.RepeatedCompositeFieldContainer[TensorsReadbackCalculatorOptions.TensorShape]
    def __init__(self, tensor_shape: _Optional[_Iterable[_Union[TensorsReadbackCalculatorOptions.TensorShape, _Mapping]]] = ...) -> None: ...
