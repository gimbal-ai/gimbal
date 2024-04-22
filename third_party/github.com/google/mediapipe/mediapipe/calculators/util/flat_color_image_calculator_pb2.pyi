from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import color_pb2 as _color_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FlatColorImageCalculatorOptions(_message.Message):
    __slots__ = ["color", "output_height", "output_width"]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_WIDTH_FIELD_NUMBER: _ClassVar[int]
    color: _color_pb2.Color
    ext: _descriptor.FieldDescriptor
    output_height: int
    output_width: int
    def __init__(self, output_width: _Optional[int] = ..., output_height: _Optional[int] = ..., color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ...) -> None: ...
