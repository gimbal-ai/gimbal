from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TrackedDetectionManagerConfig(_message.Message):
    __slots__ = ["is_same_detection_max_area_ratio", "is_same_detection_min_overlap_ratio"]
    IS_SAME_DETECTION_MAX_AREA_RATIO_FIELD_NUMBER: _ClassVar[int]
    IS_SAME_DETECTION_MIN_OVERLAP_RATIO_FIELD_NUMBER: _ClassVar[int]
    is_same_detection_max_area_ratio: float
    is_same_detection_min_overlap_ratio: float
    def __init__(self, is_same_detection_max_area_ratio: _Optional[float] = ..., is_same_detection_min_overlap_ratio: _Optional[float] = ...) -> None: ...
