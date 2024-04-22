from mediapipe.calculators.image import rotation_mode_pb2 as _rotation_mode_pb2
from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.gpu import scale_mode_pb2 as _scale_mode_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ImageTransformationCalculatorOptions(_message.Message):
    __slots__ = ["constant_padding", "flip_horizontally", "flip_vertically", "interpolation_mode", "output_height", "output_width", "padding_color", "rotation_mode", "scale_mode"]
    class InterpolationMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class Color(_message.Message):
        __slots__ = ["blue", "green", "red"]
        BLUE_FIELD_NUMBER: _ClassVar[int]
        GREEN_FIELD_NUMBER: _ClassVar[int]
        RED_FIELD_NUMBER: _ClassVar[int]
        blue: int
        green: int
        red: int
        def __init__(self, red: _Optional[int] = ..., green: _Optional[int] = ..., blue: _Optional[int] = ...) -> None: ...
    CONSTANT_PADDING_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FLIP_HORIZONTALLY_FIELD_NUMBER: _ClassVar[int]
    FLIP_VERTICALLY_FIELD_NUMBER: _ClassVar[int]
    INTERPOLATION_MODE_DEFAULT: ImageTransformationCalculatorOptions.InterpolationMode
    INTERPOLATION_MODE_FIELD_NUMBER: _ClassVar[int]
    INTERPOLATION_MODE_LINEAR: ImageTransformationCalculatorOptions.InterpolationMode
    INTERPOLATION_MODE_NEAREST: ImageTransformationCalculatorOptions.InterpolationMode
    OUTPUT_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_WIDTH_FIELD_NUMBER: _ClassVar[int]
    PADDING_COLOR_FIELD_NUMBER: _ClassVar[int]
    ROTATION_MODE_FIELD_NUMBER: _ClassVar[int]
    SCALE_MODE_FIELD_NUMBER: _ClassVar[int]
    constant_padding: bool
    ext: _descriptor.FieldDescriptor
    flip_horizontally: bool
    flip_vertically: bool
    interpolation_mode: ImageTransformationCalculatorOptions.InterpolationMode
    output_height: int
    output_width: int
    padding_color: ImageTransformationCalculatorOptions.Color
    rotation_mode: _rotation_mode_pb2.RotationMode.Mode
    scale_mode: _scale_mode_pb2.ScaleMode.Mode
    def __init__(self, output_width: _Optional[int] = ..., output_height: _Optional[int] = ..., rotation_mode: _Optional[_Union[_rotation_mode_pb2.RotationMode.Mode, str]] = ..., flip_vertically: bool = ..., flip_horizontally: bool = ..., scale_mode: _Optional[_Union[_scale_mode_pb2.ScaleMode.Mode, str]] = ..., constant_padding: bool = ..., padding_color: _Optional[_Union[ImageTransformationCalculatorOptions.Color, _Mapping]] = ..., interpolation_mode: _Optional[_Union[ImageTransformationCalculatorOptions.InterpolationMode, str]] = ...) -> None: ...
