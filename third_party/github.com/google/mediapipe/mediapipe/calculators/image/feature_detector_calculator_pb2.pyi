from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FeatureDetectorCalculatorOptions(_message.Message):
    __slots__ = ["max_features", "output_patch", "pyramid_level", "scale_factor"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MAX_FEATURES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PATCH_FIELD_NUMBER: _ClassVar[int]
    PYRAMID_LEVEL_FIELD_NUMBER: _ClassVar[int]
    SCALE_FACTOR_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    max_features: int
    output_patch: bool
    pyramid_level: int
    scale_factor: float
    def __init__(self, output_patch: bool = ..., max_features: _Optional[int] = ..., pyramid_level: _Optional[int] = ..., scale_factor: _Optional[float] = ...) -> None: ...
