from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RectToRenderScaleCalculatorOptions(_message.Message):
    __slots__ = ["multiplier", "process_timestamp_bounds"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MULTIPLIER_FIELD_NUMBER: _ClassVar[int]
    PROCESS_TIMESTAMP_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    multiplier: float
    process_timestamp_bounds: bool
    def __init__(self, multiplier: _Optional[float] = ..., process_timestamp_bounds: bool = ...) -> None: ...
