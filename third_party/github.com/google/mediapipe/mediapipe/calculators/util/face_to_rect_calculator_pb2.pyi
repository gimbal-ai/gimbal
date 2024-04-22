from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FaceToRectCalculatorOptions(_message.Message):
    __slots__ = ["eye_landmark_size", "eye_to_eye_scale", "eye_to_mouth_mix", "eye_to_mouth_scale", "mouth_landmark_size", "nose_landmark_size"]
    EYE_LANDMARK_SIZE_FIELD_NUMBER: _ClassVar[int]
    EYE_TO_EYE_SCALE_FIELD_NUMBER: _ClassVar[int]
    EYE_TO_MOUTH_MIX_FIELD_NUMBER: _ClassVar[int]
    EYE_TO_MOUTH_SCALE_FIELD_NUMBER: _ClassVar[int]
    MOUTH_LANDMARK_SIZE_FIELD_NUMBER: _ClassVar[int]
    NOSE_LANDMARK_SIZE_FIELD_NUMBER: _ClassVar[int]
    eye_landmark_size: int
    eye_to_eye_scale: float
    eye_to_mouth_mix: float
    eye_to_mouth_scale: float
    mouth_landmark_size: int
    nose_landmark_size: int
    def __init__(self, eye_landmark_size: _Optional[int] = ..., nose_landmark_size: _Optional[int] = ..., mouth_landmark_size: _Optional[int] = ..., eye_to_mouth_mix: _Optional[float] = ..., eye_to_mouth_scale: _Optional[float] = ..., eye_to_eye_scale: _Optional[float] = ...) -> None: ...
