from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SetAlphaCalculatorOptions(_message.Message):
    __slots__ = ["alpha_value"]
    ALPHA_VALUE_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    alpha_value: int
    ext: _descriptor.FieldDescriptor
    def __init__(self, alpha_value: _Optional[int] = ...) -> None: ...
