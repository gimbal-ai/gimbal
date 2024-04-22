from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AnyValue(_message.Message):
    __slots__ = ["array_value", "bool_value", "bytes_value", "double_value", "int_value", "kvlist_value", "string_value"]
    ARRAY_VALUE_FIELD_NUMBER: _ClassVar[int]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    BYTES_VALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT_VALUE_FIELD_NUMBER: _ClassVar[int]
    KVLIST_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    array_value: ArrayValue
    bool_value: bool
    bytes_value: bytes
    double_value: float
    int_value: int
    kvlist_value: KeyValueList
    string_value: str
    def __init__(self, string_value: _Optional[str] = ..., bool_value: bool = ..., int_value: _Optional[int] = ..., double_value: _Optional[float] = ..., array_value: _Optional[_Union[ArrayValue, _Mapping]] = ..., kvlist_value: _Optional[_Union[KeyValueList, _Mapping]] = ..., bytes_value: _Optional[bytes] = ...) -> None: ...

class ArrayValue(_message.Message):
    __slots__ = ["values"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[AnyValue]
    def __init__(self, values: _Optional[_Iterable[_Union[AnyValue, _Mapping]]] = ...) -> None: ...

class InstrumentationScope(_message.Message):
    __slots__ = ["attributes", "dropped_attributes_count", "name", "version"]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    DROPPED_ATTRIBUTES_COUNT_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    attributes: _containers.RepeatedCompositeFieldContainer[KeyValue]
    dropped_attributes_count: int
    name: str
    version: str
    def __init__(self, name: _Optional[str] = ..., version: _Optional[str] = ..., attributes: _Optional[_Iterable[_Union[KeyValue, _Mapping]]] = ..., dropped_attributes_count: _Optional[int] = ...) -> None: ...

class KeyValue(_message.Message):
    __slots__ = ["key", "value"]
    KEY_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    value: AnyValue
    def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[AnyValue, _Mapping]] = ...) -> None: ...

class KeyValueList(_message.Message):
    __slots__ = ["values"]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    values: _containers.RepeatedCompositeFieldContainer[KeyValue]
    def __init__(self, values: _Optional[_Iterable[_Union[KeyValue, _Mapping]]] = ...) -> None: ...
