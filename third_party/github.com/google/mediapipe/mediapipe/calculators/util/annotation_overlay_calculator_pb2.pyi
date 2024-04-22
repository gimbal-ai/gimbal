from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import color_pb2 as _color_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AnnotationOverlayCalculatorOptions(_message.Message):
    __slots__ = ["canvas_color", "canvas_height_px", "canvas_width_px", "flip_text_vertically", "gpu_scale_factor", "gpu_uses_top_left_origin"]
    CANVAS_COLOR_FIELD_NUMBER: _ClassVar[int]
    CANVAS_HEIGHT_PX_FIELD_NUMBER: _ClassVar[int]
    CANVAS_WIDTH_PX_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FLIP_TEXT_VERTICALLY_FIELD_NUMBER: _ClassVar[int]
    GPU_SCALE_FACTOR_FIELD_NUMBER: _ClassVar[int]
    GPU_USES_TOP_LEFT_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    canvas_color: _color_pb2.Color
    canvas_height_px: int
    canvas_width_px: int
    ext: _descriptor.FieldDescriptor
    flip_text_vertically: bool
    gpu_scale_factor: float
    gpu_uses_top_left_origin: bool
    def __init__(self, canvas_width_px: _Optional[int] = ..., canvas_height_px: _Optional[int] = ..., canvas_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., flip_text_vertically: bool = ..., gpu_uses_top_left_origin: bool = ..., gpu_scale_factor: _Optional[float] = ...) -> None: ...
