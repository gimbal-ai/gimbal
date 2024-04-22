from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OpticalFlowFieldData(_message.Message):
    __slots__ = ["dx", "dy", "height", "width"]
    DX_FIELD_NUMBER: _ClassVar[int]
    DY_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    dx: _containers.RepeatedScalarFieldContainer[float]
    dy: _containers.RepeatedScalarFieldContainer[float]
    height: int
    width: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., dx: _Optional[_Iterable[float]] = ..., dy: _Optional[_Iterable[float]] = ...) -> None: ...
