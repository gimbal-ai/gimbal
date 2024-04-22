from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import color_pb2 as _color_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TimedBoxListToRenderDataCalculatorOptions(_message.Message):
    __slots__ = ["box_color", "thickness"]
    BOX_COLOR_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    THICKNESS_FIELD_NUMBER: _ClassVar[int]
    box_color: _color_pb2.Color
    ext: _descriptor.FieldDescriptor
    thickness: float
    def __init__(self, box_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., thickness: _Optional[float] = ...) -> None: ...
