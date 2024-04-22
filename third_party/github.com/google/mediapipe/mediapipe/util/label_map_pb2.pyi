from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class LabelMapItem(_message.Message):
    __slots__ = ["child_name", "display_name", "name"]
    CHILD_NAME_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    child_name: _containers.RepeatedScalarFieldContainer[str]
    display_name: str
    name: str
    def __init__(self, name: _Optional[str] = ..., display_name: _Optional[str] = ..., child_name: _Optional[_Iterable[str]] = ...) -> None: ...
