from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ThresholdingCalculatorOptions(_message.Message):
    __slots__ = ["threshold"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    threshold: float
    def __init__(self, threshold: _Optional[float] = ...) -> None: ...
