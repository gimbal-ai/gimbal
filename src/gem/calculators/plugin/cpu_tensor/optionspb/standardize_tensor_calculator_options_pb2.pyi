from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class StandardizeTensorCalculatorOptions(_message.Message):
    __slots__ = ["mean", "stddev"]
    MEAN_FIELD_NUMBER: _ClassVar[int]
    STDDEV_FIELD_NUMBER: _ClassVar[int]
    mean: _containers.RepeatedScalarFieldContainer[float]
    stddev: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, mean: _Optional[_Iterable[float]] = ..., stddev: _Optional[_Iterable[float]] = ...) -> None: ...
