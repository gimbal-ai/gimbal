from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.gpu import scale_mode_pb2 as _scale_mode_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GlScalerCalculatorOptions(_message.Message):
    __slots__ = ["flip_horizontal", "flip_vertical", "output_height", "output_scale", "output_width", "rotation", "scale_mode", "use_nearest_neighbor_interpolation"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FLIP_HORIZONTAL_FIELD_NUMBER: _ClassVar[int]
    FLIP_VERTICAL_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SCALE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_WIDTH_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    SCALE_MODE_FIELD_NUMBER: _ClassVar[int]
    USE_NEAREST_NEIGHBOR_INTERPOLATION_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    flip_horizontal: bool
    flip_vertical: bool
    output_height: int
    output_scale: float
    output_width: int
    rotation: int
    scale_mode: _scale_mode_pb2.ScaleMode.Mode
    use_nearest_neighbor_interpolation: bool
    def __init__(self, output_width: _Optional[int] = ..., output_height: _Optional[int] = ..., output_scale: _Optional[float] = ..., rotation: _Optional[int] = ..., flip_vertical: bool = ..., flip_horizontal: bool = ..., scale_mode: _Optional[_Union[_scale_mode_pb2.ScaleMode.Mode, str]] = ..., use_nearest_neighbor_interpolation: bool = ...) -> None: ...
