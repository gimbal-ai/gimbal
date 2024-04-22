from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TensorRTExecuteCalculatorOptions(_message.Message):
    __slots__ = ["input_onnx_name", "output_onnx_name"]
    INPUT_ONNX_NAME_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ONNX_NAME_FIELD_NUMBER: _ClassVar[int]
    input_onnx_name: _containers.RepeatedScalarFieldContainer[str]
    output_onnx_name: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, input_onnx_name: _Optional[_Iterable[str]] = ..., output_onnx_name: _Optional[_Iterable[str]] = ...) -> None: ...
