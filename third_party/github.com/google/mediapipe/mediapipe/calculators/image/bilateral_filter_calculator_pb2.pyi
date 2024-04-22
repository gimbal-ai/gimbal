from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class BilateralFilterCalculatorOptions(_message.Message):
    __slots__ = ["sigma_color", "sigma_space"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    SIGMA_COLOR_FIELD_NUMBER: _ClassVar[int]
    SIGMA_SPACE_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    sigma_color: float
    sigma_space: float
    def __init__(self, sigma_color: _Optional[float] = ..., sigma_space: _Optional[float] = ...) -> None: ...
