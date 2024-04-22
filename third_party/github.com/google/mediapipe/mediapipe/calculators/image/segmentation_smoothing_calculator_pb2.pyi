from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SegmentationSmoothingCalculatorOptions(_message.Message):
    __slots__ = ["combine_with_previous_ratio"]
    COMBINE_WITH_PREVIOUS_RATIO_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    combine_with_previous_ratio: float
    ext: _descriptor.FieldDescriptor
    def __init__(self, combine_with_previous_ratio: _Optional[float] = ...) -> None: ...
