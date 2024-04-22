from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GlAnimationOverlayCalculatorOptions(_message.Message):
    __slots__ = ["animation_speed_fps", "aspect_ratio", "vertical_fov_degrees", "z_clipping_plane_far", "z_clipping_plane_near"]
    ANIMATION_SPEED_FPS_FIELD_NUMBER: _ClassVar[int]
    ASPECT_RATIO_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_FOV_DEGREES_FIELD_NUMBER: _ClassVar[int]
    Z_CLIPPING_PLANE_FAR_FIELD_NUMBER: _ClassVar[int]
    Z_CLIPPING_PLANE_NEAR_FIELD_NUMBER: _ClassVar[int]
    animation_speed_fps: float
    aspect_ratio: float
    ext: _descriptor.FieldDescriptor
    vertical_fov_degrees: float
    z_clipping_plane_far: float
    z_clipping_plane_near: float
    def __init__(self, aspect_ratio: _Optional[float] = ..., vertical_fov_degrees: _Optional[float] = ..., z_clipping_plane_near: _Optional[float] = ..., z_clipping_plane_far: _Optional[float] = ..., animation_speed_fps: _Optional[float] = ...) -> None: ...
