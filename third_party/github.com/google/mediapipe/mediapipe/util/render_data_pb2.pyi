from mediapipe.util import color_pb2 as _color_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RenderAnnotation(_message.Message):
    __slots__ = ["arrow", "color", "filled_oval", "filled_rectangle", "filled_rounded_rectangle", "gradient_line", "line", "oval", "point", "rectangle", "rounded_rectangle", "scene_tag", "scribble", "text", "thickness"]
    class Arrow(_message.Message):
        __slots__ = ["normalized", "x_end", "x_start", "y_end", "y_start"]
        NORMALIZED_FIELD_NUMBER: _ClassVar[int]
        X_END_FIELD_NUMBER: _ClassVar[int]
        X_START_FIELD_NUMBER: _ClassVar[int]
        Y_END_FIELD_NUMBER: _ClassVar[int]
        Y_START_FIELD_NUMBER: _ClassVar[int]
        normalized: bool
        x_end: float
        x_start: float
        y_end: float
        y_start: float
        def __init__(self, x_start: _Optional[float] = ..., y_start: _Optional[float] = ..., x_end: _Optional[float] = ..., y_end: _Optional[float] = ..., normalized: bool = ...) -> None: ...
    class FilledOval(_message.Message):
        __slots__ = ["fill_color", "oval"]
        FILL_COLOR_FIELD_NUMBER: _ClassVar[int]
        OVAL_FIELD_NUMBER: _ClassVar[int]
        fill_color: _color_pb2.Color
        oval: RenderAnnotation.Oval
        def __init__(self, oval: _Optional[_Union[RenderAnnotation.Oval, _Mapping]] = ..., fill_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ...) -> None: ...
    class FilledRectangle(_message.Message):
        __slots__ = ["fill_color", "rectangle"]
        FILL_COLOR_FIELD_NUMBER: _ClassVar[int]
        RECTANGLE_FIELD_NUMBER: _ClassVar[int]
        fill_color: _color_pb2.Color
        rectangle: RenderAnnotation.Rectangle
        def __init__(self, rectangle: _Optional[_Union[RenderAnnotation.Rectangle, _Mapping]] = ..., fill_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ...) -> None: ...
    class FilledRoundedRectangle(_message.Message):
        __slots__ = ["fill_color", "rounded_rectangle"]
        FILL_COLOR_FIELD_NUMBER: _ClassVar[int]
        ROUNDED_RECTANGLE_FIELD_NUMBER: _ClassVar[int]
        fill_color: _color_pb2.Color
        rounded_rectangle: RenderAnnotation.RoundedRectangle
        def __init__(self, rounded_rectangle: _Optional[_Union[RenderAnnotation.RoundedRectangle, _Mapping]] = ..., fill_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ...) -> None: ...
    class GradientLine(_message.Message):
        __slots__ = ["color1", "color2", "normalized", "x_end", "x_start", "y_end", "y_start"]
        COLOR1_FIELD_NUMBER: _ClassVar[int]
        COLOR2_FIELD_NUMBER: _ClassVar[int]
        NORMALIZED_FIELD_NUMBER: _ClassVar[int]
        X_END_FIELD_NUMBER: _ClassVar[int]
        X_START_FIELD_NUMBER: _ClassVar[int]
        Y_END_FIELD_NUMBER: _ClassVar[int]
        Y_START_FIELD_NUMBER: _ClassVar[int]
        color1: _color_pb2.Color
        color2: _color_pb2.Color
        normalized: bool
        x_end: float
        x_start: float
        y_end: float
        y_start: float
        def __init__(self, x_start: _Optional[float] = ..., y_start: _Optional[float] = ..., x_end: _Optional[float] = ..., y_end: _Optional[float] = ..., normalized: bool = ..., color1: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., color2: _Optional[_Union[_color_pb2.Color, _Mapping]] = ...) -> None: ...
    class Line(_message.Message):
        __slots__ = ["line_type", "normalized", "x_end", "x_start", "y_end", "y_start"]
        class LineType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        DASHED: RenderAnnotation.Line.LineType
        LINE_TYPE_FIELD_NUMBER: _ClassVar[int]
        NORMALIZED_FIELD_NUMBER: _ClassVar[int]
        SOLID: RenderAnnotation.Line.LineType
        UNKNOWN: RenderAnnotation.Line.LineType
        X_END_FIELD_NUMBER: _ClassVar[int]
        X_START_FIELD_NUMBER: _ClassVar[int]
        Y_END_FIELD_NUMBER: _ClassVar[int]
        Y_START_FIELD_NUMBER: _ClassVar[int]
        line_type: RenderAnnotation.Line.LineType
        normalized: bool
        x_end: float
        x_start: float
        y_end: float
        y_start: float
        def __init__(self, x_start: _Optional[float] = ..., y_start: _Optional[float] = ..., x_end: _Optional[float] = ..., y_end: _Optional[float] = ..., normalized: bool = ..., line_type: _Optional[_Union[RenderAnnotation.Line.LineType, str]] = ...) -> None: ...
    class Oval(_message.Message):
        __slots__ = ["rectangle"]
        RECTANGLE_FIELD_NUMBER: _ClassVar[int]
        rectangle: RenderAnnotation.Rectangle
        def __init__(self, rectangle: _Optional[_Union[RenderAnnotation.Rectangle, _Mapping]] = ...) -> None: ...
    class Point(_message.Message):
        __slots__ = ["normalized", "x", "y"]
        NORMALIZED_FIELD_NUMBER: _ClassVar[int]
        X_FIELD_NUMBER: _ClassVar[int]
        Y_FIELD_NUMBER: _ClassVar[int]
        normalized: bool
        x: float
        y: float
        def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., normalized: bool = ...) -> None: ...
    class Rectangle(_message.Message):
        __slots__ = ["bottom", "left", "normalized", "right", "rotation", "top", "top_left_thickness"]
        BOTTOM_FIELD_NUMBER: _ClassVar[int]
        LEFT_FIELD_NUMBER: _ClassVar[int]
        NORMALIZED_FIELD_NUMBER: _ClassVar[int]
        RIGHT_FIELD_NUMBER: _ClassVar[int]
        ROTATION_FIELD_NUMBER: _ClassVar[int]
        TOP_FIELD_NUMBER: _ClassVar[int]
        TOP_LEFT_THICKNESS_FIELD_NUMBER: _ClassVar[int]
        bottom: float
        left: float
        normalized: bool
        right: float
        rotation: float
        top: float
        top_left_thickness: float
        def __init__(self, left: _Optional[float] = ..., top: _Optional[float] = ..., right: _Optional[float] = ..., bottom: _Optional[float] = ..., normalized: bool = ..., rotation: _Optional[float] = ..., top_left_thickness: _Optional[float] = ...) -> None: ...
    class RoundedRectangle(_message.Message):
        __slots__ = ["corner_radius", "line_type", "rectangle"]
        CORNER_RADIUS_FIELD_NUMBER: _ClassVar[int]
        LINE_TYPE_FIELD_NUMBER: _ClassVar[int]
        RECTANGLE_FIELD_NUMBER: _ClassVar[int]
        corner_radius: int
        line_type: int
        rectangle: RenderAnnotation.Rectangle
        def __init__(self, rectangle: _Optional[_Union[RenderAnnotation.Rectangle, _Mapping]] = ..., corner_radius: _Optional[int] = ..., line_type: _Optional[int] = ...) -> None: ...
    class Scribble(_message.Message):
        __slots__ = ["point"]
        POINT_FIELD_NUMBER: _ClassVar[int]
        point: _containers.RepeatedCompositeFieldContainer[RenderAnnotation.Point]
        def __init__(self, point: _Optional[_Iterable[_Union[RenderAnnotation.Point, _Mapping]]] = ...) -> None: ...
    class Text(_message.Message):
        __slots__ = ["baseline", "center_horizontally", "center_vertically", "display_text", "font_face", "font_height", "left", "normalized", "outline_color", "outline_thickness"]
        BASELINE_FIELD_NUMBER: _ClassVar[int]
        CENTER_HORIZONTALLY_FIELD_NUMBER: _ClassVar[int]
        CENTER_VERTICALLY_FIELD_NUMBER: _ClassVar[int]
        DISPLAY_TEXT_FIELD_NUMBER: _ClassVar[int]
        FONT_FACE_FIELD_NUMBER: _ClassVar[int]
        FONT_HEIGHT_FIELD_NUMBER: _ClassVar[int]
        LEFT_FIELD_NUMBER: _ClassVar[int]
        NORMALIZED_FIELD_NUMBER: _ClassVar[int]
        OUTLINE_COLOR_FIELD_NUMBER: _ClassVar[int]
        OUTLINE_THICKNESS_FIELD_NUMBER: _ClassVar[int]
        baseline: float
        center_horizontally: bool
        center_vertically: bool
        display_text: str
        font_face: int
        font_height: float
        left: float
        normalized: bool
        outline_color: _color_pb2.Color
        outline_thickness: float
        def __init__(self, display_text: _Optional[str] = ..., left: _Optional[float] = ..., baseline: _Optional[float] = ..., font_height: _Optional[float] = ..., normalized: bool = ..., font_face: _Optional[int] = ..., center_horizontally: bool = ..., center_vertically: bool = ..., outline_thickness: _Optional[float] = ..., outline_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ...) -> None: ...
    ARROW_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    FILLED_OVAL_FIELD_NUMBER: _ClassVar[int]
    FILLED_RECTANGLE_FIELD_NUMBER: _ClassVar[int]
    FILLED_ROUNDED_RECTANGLE_FIELD_NUMBER: _ClassVar[int]
    GRADIENT_LINE_FIELD_NUMBER: _ClassVar[int]
    LINE_FIELD_NUMBER: _ClassVar[int]
    OVAL_FIELD_NUMBER: _ClassVar[int]
    POINT_FIELD_NUMBER: _ClassVar[int]
    RECTANGLE_FIELD_NUMBER: _ClassVar[int]
    ROUNDED_RECTANGLE_FIELD_NUMBER: _ClassVar[int]
    SCENE_TAG_FIELD_NUMBER: _ClassVar[int]
    SCRIBBLE_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    THICKNESS_FIELD_NUMBER: _ClassVar[int]
    arrow: RenderAnnotation.Arrow
    color: _color_pb2.Color
    filled_oval: RenderAnnotation.FilledOval
    filled_rectangle: RenderAnnotation.FilledRectangle
    filled_rounded_rectangle: RenderAnnotation.FilledRoundedRectangle
    gradient_line: RenderAnnotation.GradientLine
    line: RenderAnnotation.Line
    oval: RenderAnnotation.Oval
    point: RenderAnnotation.Point
    rectangle: RenderAnnotation.Rectangle
    rounded_rectangle: RenderAnnotation.RoundedRectangle
    scene_tag: str
    scribble: RenderAnnotation.Scribble
    text: RenderAnnotation.Text
    thickness: float
    def __init__(self, rectangle: _Optional[_Union[RenderAnnotation.Rectangle, _Mapping]] = ..., filled_rectangle: _Optional[_Union[RenderAnnotation.FilledRectangle, _Mapping]] = ..., oval: _Optional[_Union[RenderAnnotation.Oval, _Mapping]] = ..., filled_oval: _Optional[_Union[RenderAnnotation.FilledOval, _Mapping]] = ..., point: _Optional[_Union[RenderAnnotation.Point, _Mapping]] = ..., line: _Optional[_Union[RenderAnnotation.Line, _Mapping]] = ..., arrow: _Optional[_Union[RenderAnnotation.Arrow, _Mapping]] = ..., text: _Optional[_Union[RenderAnnotation.Text, _Mapping]] = ..., rounded_rectangle: _Optional[_Union[RenderAnnotation.RoundedRectangle, _Mapping]] = ..., filled_rounded_rectangle: _Optional[_Union[RenderAnnotation.FilledRoundedRectangle, _Mapping]] = ..., gradient_line: _Optional[_Union[RenderAnnotation.GradientLine, _Mapping]] = ..., scribble: _Optional[_Union[RenderAnnotation.Scribble, _Mapping]] = ..., thickness: _Optional[float] = ..., color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., scene_tag: _Optional[str] = ...) -> None: ...

class RenderData(_message.Message):
    __slots__ = ["render_annotations", "scene_class", "scene_viewport"]
    RENDER_ANNOTATIONS_FIELD_NUMBER: _ClassVar[int]
    SCENE_CLASS_FIELD_NUMBER: _ClassVar[int]
    SCENE_VIEWPORT_FIELD_NUMBER: _ClassVar[int]
    render_annotations: _containers.RepeatedCompositeFieldContainer[RenderAnnotation]
    scene_class: str
    scene_viewport: RenderViewport
    def __init__(self, render_annotations: _Optional[_Iterable[_Union[RenderAnnotation, _Mapping]]] = ..., scene_class: _Optional[str] = ..., scene_viewport: _Optional[_Union[RenderViewport, _Mapping]] = ...) -> None: ...

class RenderViewport(_message.Message):
    __slots__ = ["compose_on_video", "height_px", "id", "width_px"]
    COMPOSE_ON_VIDEO_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_PX_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    WIDTH_PX_FIELD_NUMBER: _ClassVar[int]
    compose_on_video: bool
    height_px: int
    id: str
    width_px: int
    def __init__(self, id: _Optional[str] = ..., width_px: _Optional[int] = ..., height_px: _Optional[int] = ..., compose_on_video: bool = ...) -> None: ...
