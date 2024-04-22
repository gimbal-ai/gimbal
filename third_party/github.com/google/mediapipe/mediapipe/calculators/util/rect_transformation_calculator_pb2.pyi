from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RectTransformationCalculatorOptions(_message.Message):
    __slots__ = ["rotation", "rotation_degrees", "scale_x", "scale_y", "shift_x", "shift_y", "square_long", "square_short"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    ROTATION_DEGREES_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    SCALE_X_FIELD_NUMBER: _ClassVar[int]
    SCALE_Y_FIELD_NUMBER: _ClassVar[int]
    SHIFT_X_FIELD_NUMBER: _ClassVar[int]
    SHIFT_Y_FIELD_NUMBER: _ClassVar[int]
    SQUARE_LONG_FIELD_NUMBER: _ClassVar[int]
    SQUARE_SHORT_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    rotation: float
    rotation_degrees: int
    scale_x: float
    scale_y: float
    shift_x: float
    shift_y: float
    square_long: bool
    square_short: bool
    def __init__(self, scale_x: _Optional[float] = ..., scale_y: _Optional[float] = ..., rotation: _Optional[float] = ..., rotation_degrees: _Optional[int] = ..., shift_x: _Optional[float] = ..., shift_y: _Optional[float] = ..., square_long: bool = ..., square_short: bool = ...) -> None: ...
