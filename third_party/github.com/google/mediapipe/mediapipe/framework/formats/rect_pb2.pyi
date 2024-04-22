from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class NormalizedRect(_message.Message):
    __slots__ = ["height", "rect_id", "rotation", "width", "x_center", "y_center"]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    RECT_ID_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    X_CENTER_FIELD_NUMBER: _ClassVar[int]
    Y_CENTER_FIELD_NUMBER: _ClassVar[int]
    height: float
    rect_id: int
    rotation: float
    width: float
    x_center: float
    y_center: float
    def __init__(self, x_center: _Optional[float] = ..., y_center: _Optional[float] = ..., height: _Optional[float] = ..., width: _Optional[float] = ..., rotation: _Optional[float] = ..., rect_id: _Optional[int] = ...) -> None: ...

class Rect(_message.Message):
    __slots__ = ["height", "rect_id", "rotation", "width", "x_center", "y_center"]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    RECT_ID_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    X_CENTER_FIELD_NUMBER: _ClassVar[int]
    Y_CENTER_FIELD_NUMBER: _ClassVar[int]
    height: int
    rect_id: int
    rotation: float
    width: int
    x_center: int
    y_center: int
    def __init__(self, x_center: _Optional[int] = ..., y_center: _Optional[int] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., rotation: _Optional[float] = ..., rect_id: _Optional[int] = ...) -> None: ...
