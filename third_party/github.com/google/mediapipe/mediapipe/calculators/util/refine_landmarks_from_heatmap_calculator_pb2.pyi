from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RefineLandmarksFromHeatmapCalculatorOptions(_message.Message):
    __slots__ = ["kernel_size", "min_confidence_to_refine", "refine_presence", "refine_visibility"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    KERNEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    MIN_CONFIDENCE_TO_REFINE_FIELD_NUMBER: _ClassVar[int]
    REFINE_PRESENCE_FIELD_NUMBER: _ClassVar[int]
    REFINE_VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    kernel_size: int
    min_confidence_to_refine: float
    refine_presence: bool
    refine_visibility: bool
    def __init__(self, kernel_size: _Optional[int] = ..., min_confidence_to_refine: _Optional[float] = ..., refine_presence: bool = ..., refine_visibility: bool = ...) -> None: ...
