from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import color_pb2 as _color_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LabelsToRenderDataCalculatorOptions(_message.Message):
    __slots__ = ["color", "display_classification_score", "font_face", "font_height_px", "horizontal_offset_px", "location", "max_num_labels", "outline_color", "outline_thickness", "thickness", "use_display_name", "vertical_offset_px"]
    class Location(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BOTTOM_LEFT: LabelsToRenderDataCalculatorOptions.Location
    COLOR_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_CLASSIFICATION_SCORE_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FONT_FACE_FIELD_NUMBER: _ClassVar[int]
    FONT_HEIGHT_PX_FIELD_NUMBER: _ClassVar[int]
    HORIZONTAL_OFFSET_PX_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    MAX_NUM_LABELS_FIELD_NUMBER: _ClassVar[int]
    OUTLINE_COLOR_FIELD_NUMBER: _ClassVar[int]
    OUTLINE_THICKNESS_FIELD_NUMBER: _ClassVar[int]
    THICKNESS_FIELD_NUMBER: _ClassVar[int]
    TOP_LEFT: LabelsToRenderDataCalculatorOptions.Location
    USE_DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    VERTICAL_OFFSET_PX_FIELD_NUMBER: _ClassVar[int]
    color: _containers.RepeatedCompositeFieldContainer[_color_pb2.Color]
    display_classification_score: bool
    ext: _descriptor.FieldDescriptor
    font_face: int
    font_height_px: int
    horizontal_offset_px: int
    location: LabelsToRenderDataCalculatorOptions.Location
    max_num_labels: int
    outline_color: _containers.RepeatedCompositeFieldContainer[_color_pb2.Color]
    outline_thickness: float
    thickness: float
    use_display_name: bool
    vertical_offset_px: int
    def __init__(self, color: _Optional[_Iterable[_Union[_color_pb2.Color, _Mapping]]] = ..., thickness: _Optional[float] = ..., outline_color: _Optional[_Iterable[_Union[_color_pb2.Color, _Mapping]]] = ..., outline_thickness: _Optional[float] = ..., font_height_px: _Optional[int] = ..., horizontal_offset_px: _Optional[int] = ..., vertical_offset_px: _Optional[int] = ..., max_num_labels: _Optional[int] = ..., font_face: _Optional[int] = ..., location: _Optional[_Union[LabelsToRenderDataCalculatorOptions.Location, str]] = ..., use_display_name: bool = ..., display_classification_score: bool = ...) -> None: ...
