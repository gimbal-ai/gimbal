from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class InputOnlyProto(_message.Message):
    __slots__ = ["x"]
    X_FIELD_NUMBER: _ClassVar[int]
    x: int
    def __init__(self, x: _Optional[int] = ...) -> None: ...

class PacketTestProto(_message.Message):
    __slots__ = ["x", "y"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: _containers.RepeatedScalarFieldContainer[int]
    y: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, x: _Optional[_Iterable[int]] = ..., y: _Optional[_Iterable[int]] = ...) -> None: ...

class SerializationProxyProto(_message.Message):
    __slots__ = ["bool_value", "float_value", "string_value"]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    bool_value: bool
    float_value: _containers.RepeatedScalarFieldContainer[float]
    string_value: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, bool_value: bool = ..., float_value: _Optional[_Iterable[float]] = ..., string_value: _Optional[_Iterable[str]] = ...) -> None: ...

class SimpleProto(_message.Message):
    __slots__ = ["value"]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    value: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, value: _Optional[_Iterable[bytes]] = ...) -> None: ...
