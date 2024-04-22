from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class StabilizedLogCalculatorOptions(_message.Message):
    __slots__ = ["check_nonnegativity", "output_scale", "stabilizer"]
    CHECK_NONNEGATIVITY_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SCALE_FIELD_NUMBER: _ClassVar[int]
    STABILIZER_FIELD_NUMBER: _ClassVar[int]
    check_nonnegativity: bool
    ext: _descriptor.FieldDescriptor
    output_scale: float
    stabilizer: float
    def __init__(self, stabilizer: _Optional[float] = ..., check_nonnegativity: bool = ..., output_scale: _Optional[float] = ...) -> None: ...
