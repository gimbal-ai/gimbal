from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import color_pb2 as _color_pb2
from mediapipe.util import render_data_pb2 as _render_data_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DetectionsToRenderDataCalculatorOptions(_message.Message):
    __slots__ = ["color", "one_label_per_line", "produce_empty_packet", "render_detection_id", "scene_class", "text", "text_delimiter", "thickness"]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    ONE_LABEL_PER_LINE_FIELD_NUMBER: _ClassVar[int]
    PRODUCE_EMPTY_PACKET_FIELD_NUMBER: _ClassVar[int]
    RENDER_DETECTION_ID_FIELD_NUMBER: _ClassVar[int]
    SCENE_CLASS_FIELD_NUMBER: _ClassVar[int]
    TEXT_DELIMITER_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    THICKNESS_FIELD_NUMBER: _ClassVar[int]
    color: _color_pb2.Color
    ext: _descriptor.FieldDescriptor
    one_label_per_line: bool
    produce_empty_packet: bool
    render_detection_id: bool
    scene_class: str
    text: _render_data_pb2.RenderAnnotation.Text
    text_delimiter: str
    thickness: float
    def __init__(self, produce_empty_packet: bool = ..., text_delimiter: _Optional[str] = ..., one_label_per_line: bool = ..., text: _Optional[_Union[_render_data_pb2.RenderAnnotation.Text, _Mapping]] = ..., thickness: _Optional[float] = ..., color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., scene_class: _Optional[str] = ..., render_detection_id: bool = ...) -> None: ...
