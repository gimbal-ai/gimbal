from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FieldData(_message.Message):
    __slots__ = ["bool_value", "double_value", "enum_value", "float_value", "int32_value", "int64_value", "message_value", "string_value", "uint32_value", "uint64_value"]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    ENUM_VALUE_FIELD_NUMBER: _ClassVar[int]
    FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT32_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT64_VALUE_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    UINT32_VALUE_FIELD_NUMBER: _ClassVar[int]
    UINT64_VALUE_FIELD_NUMBER: _ClassVar[int]
    bool_value: bool
    double_value: float
    enum_value: int
    float_value: float
    int32_value: int
    int64_value: int
    message_value: MessageData
    string_value: str
    uint32_value: int
    uint64_value: int
    def __init__(self, int32_value: _Optional[int] = ..., int64_value: _Optional[int] = ..., uint32_value: _Optional[int] = ..., uint64_value: _Optional[int] = ..., double_value: _Optional[float] = ..., float_value: _Optional[float] = ..., bool_value: bool = ..., enum_value: _Optional[int] = ..., string_value: _Optional[str] = ..., message_value: _Optional[_Union[MessageData, _Mapping]] = ...) -> None: ...

class MessageData(_message.Message):
    __slots__ = ["type_url", "value"]
    TYPE_URL_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    type_url: str
    value: bytes
    def __init__(self, type_url: _Optional[str] = ..., value: _Optional[bytes] = ...) -> None: ...
