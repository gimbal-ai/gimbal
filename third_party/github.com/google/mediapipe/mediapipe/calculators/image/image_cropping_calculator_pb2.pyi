from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ImageCroppingCalculatorOptions(_message.Message):
    __slots__ = ["border_mode", "height", "norm_center_x", "norm_center_y", "norm_height", "norm_width", "output_max_height", "output_max_width", "rotation", "width"]
    class BorderMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BORDER_MODE_FIELD_NUMBER: _ClassVar[int]
    CROP_BORDER_REPLICATE: ImageCroppingCalculatorOptions.BorderMode
    CROP_BORDER_UNSPECIFIED: ImageCroppingCalculatorOptions.BorderMode
    CROP_BORDER_ZERO: ImageCroppingCalculatorOptions.BorderMode
    EXT_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    NORM_CENTER_X_FIELD_NUMBER: _ClassVar[int]
    NORM_CENTER_Y_FIELD_NUMBER: _ClassVar[int]
    NORM_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    NORM_WIDTH_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MAX_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MAX_WIDTH_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    border_mode: ImageCroppingCalculatorOptions.BorderMode
    ext: _descriptor.FieldDescriptor
    height: int
    norm_center_x: float
    norm_center_y: float
    norm_height: float
    norm_width: float
    output_max_height: int
    output_max_width: int
    rotation: float
    width: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., rotation: _Optional[float] = ..., norm_width: _Optional[float] = ..., norm_height: _Optional[float] = ..., norm_center_x: _Optional[float] = ..., norm_center_y: _Optional[float] = ..., border_mode: _Optional[_Union[ImageCroppingCalculatorOptions.BorderMode, str]] = ..., output_max_width: _Optional[int] = ..., output_max_height: _Optional[int] = ...) -> None: ...
