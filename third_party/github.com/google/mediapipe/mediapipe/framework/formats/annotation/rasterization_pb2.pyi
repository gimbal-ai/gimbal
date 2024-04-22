from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Rasterization(_message.Message):
    __slots__ = ["interval"]
    class Interval(_message.Message):
        __slots__ = ["left_x", "right_x", "y"]
        LEFT_X_FIELD_NUMBER: _ClassVar[int]
        RIGHT_X_FIELD_NUMBER: _ClassVar[int]
        Y_FIELD_NUMBER: _ClassVar[int]
        left_x: int
        right_x: int
        y: int
        def __init__(self, y: _Optional[int] = ..., left_x: _Optional[int] = ..., right_x: _Optional[int] = ...) -> None: ...
    INTERVAL_FIELD_NUMBER: _ClassVar[int]
    interval: _containers.RepeatedCompositeFieldContainer[Rasterization.Interval]
    def __init__(self, interval: _Optional[_Iterable[_Union[Rasterization.Interval, _Mapping]]] = ...) -> None: ...
