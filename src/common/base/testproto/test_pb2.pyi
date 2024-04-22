from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TestChildMessage(_message.Message):
    __slots__ = ["string_val"]
    STRING_VAL_FIELD_NUMBER: _ClassVar[int]
    string_val: str
    def __init__(self, string_val: _Optional[str] = ...) -> None: ...

class TestParentMessage(_message.Message):
    __slots__ = ["child", "int_val"]
    CHILD_FIELD_NUMBER: _ClassVar[int]
    INT_VAL_FIELD_NUMBER: _ClassVar[int]
    child: _containers.RepeatedCompositeFieldContainer[TestChildMessage]
    int_val: int
    def __init__(self, int_val: _Optional[int] = ..., child: _Optional[_Iterable[_Union[TestChildMessage, _Mapping]]] = ...) -> None: ...
