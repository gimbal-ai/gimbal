from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class BypassCalculatorOptions(_message.Message):
    __slots__ = ["pass_input_stream", "pass_output_stream"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    PASS_INPUT_STREAM_FIELD_NUMBER: _ClassVar[int]
    PASS_OUTPUT_STREAM_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    pass_input_stream: _containers.RepeatedScalarFieldContainer[str]
    pass_output_stream: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, pass_input_stream: _Optional[_Iterable[str]] = ..., pass_output_stream: _Optional[_Iterable[str]] = ...) -> None: ...
