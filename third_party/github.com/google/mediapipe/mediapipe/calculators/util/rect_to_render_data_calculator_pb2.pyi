from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import color_pb2 as _color_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RectToRenderDataCalculatorOptions(_message.Message):
    __slots__ = ["color", "filled", "oval", "thickness", "top_left_thickness"]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FILLED_FIELD_NUMBER: _ClassVar[int]
    OVAL_FIELD_NUMBER: _ClassVar[int]
    THICKNESS_FIELD_NUMBER: _ClassVar[int]
    TOP_LEFT_THICKNESS_FIELD_NUMBER: _ClassVar[int]
    color: _color_pb2.Color
    ext: _descriptor.FieldDescriptor
    filled: bool
    oval: bool
    thickness: float
    top_left_thickness: float
    def __init__(self, filled: bool = ..., color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., thickness: _Optional[float] = ..., oval: bool = ..., top_left_thickness: _Optional[float] = ...) -> None: ...
