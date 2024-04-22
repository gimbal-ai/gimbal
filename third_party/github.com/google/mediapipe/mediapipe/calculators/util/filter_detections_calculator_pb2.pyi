from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FilterDetectionsCalculatorOptions(_message.Message):
    __slots__ = ["max_pixel_size", "min_pixel_size", "min_score"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MAX_PIXEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    MIN_PIXEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    MIN_SCORE_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    max_pixel_size: float
    min_pixel_size: float
    min_score: float
    def __init__(self, min_score: _Optional[float] = ..., min_pixel_size: _Optional[float] = ..., max_pixel_size: _Optional[float] = ...) -> None: ...
