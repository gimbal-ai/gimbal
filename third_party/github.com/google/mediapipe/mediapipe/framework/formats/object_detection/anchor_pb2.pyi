from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class Anchor(_message.Message):
    __slots__ = ["h", "w", "x_center", "y_center"]
    H_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    X_CENTER_FIELD_NUMBER: _ClassVar[int]
    Y_CENTER_FIELD_NUMBER: _ClassVar[int]
    h: float
    w: float
    x_center: float
    y_center: float
    def __init__(self, x_center: _Optional[float] = ..., y_center: _Optional[float] = ..., h: _Optional[float] = ..., w: _Optional[float] = ...) -> None: ...
