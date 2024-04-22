from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import color_pb2 as _color_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LandmarksToRenderDataCalculatorOptions(_message.Message):
    __slots__ = ["connection_color", "landmark_color", "landmark_connections", "max_depth_circle_thickness", "max_depth_line_color", "min_depth_circle_thickness", "min_depth_line_color", "presence_threshold", "render_landmarks", "thickness", "utilize_presence", "utilize_visibility", "visibility_threshold", "visualize_landmark_depth"]
    CONNECTION_COLOR_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    LANDMARK_COLOR_FIELD_NUMBER: _ClassVar[int]
    LANDMARK_CONNECTIONS_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_CIRCLE_THICKNESS_FIELD_NUMBER: _ClassVar[int]
    MAX_DEPTH_LINE_COLOR_FIELD_NUMBER: _ClassVar[int]
    MIN_DEPTH_CIRCLE_THICKNESS_FIELD_NUMBER: _ClassVar[int]
    MIN_DEPTH_LINE_COLOR_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    RENDER_LANDMARKS_FIELD_NUMBER: _ClassVar[int]
    THICKNESS_FIELD_NUMBER: _ClassVar[int]
    UTILIZE_PRESENCE_FIELD_NUMBER: _ClassVar[int]
    UTILIZE_VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    VISUALIZE_LANDMARK_DEPTH_FIELD_NUMBER: _ClassVar[int]
    connection_color: _color_pb2.Color
    ext: _descriptor.FieldDescriptor
    landmark_color: _color_pb2.Color
    landmark_connections: _containers.RepeatedScalarFieldContainer[int]
    max_depth_circle_thickness: float
    max_depth_line_color: _color_pb2.Color
    min_depth_circle_thickness: float
    min_depth_line_color: _color_pb2.Color
    presence_threshold: float
    render_landmarks: bool
    thickness: float
    utilize_presence: bool
    utilize_visibility: bool
    visibility_threshold: float
    visualize_landmark_depth: bool
    def __init__(self, landmark_connections: _Optional[_Iterable[int]] = ..., landmark_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., render_landmarks: bool = ..., connection_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., thickness: _Optional[float] = ..., visualize_landmark_depth: bool = ..., utilize_visibility: bool = ..., visibility_threshold: _Optional[float] = ..., utilize_presence: bool = ..., presence_threshold: _Optional[float] = ..., min_depth_circle_thickness: _Optional[float] = ..., max_depth_circle_thickness: _Optional[float] = ..., min_depth_line_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., max_depth_line_color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ...) -> None: ...
