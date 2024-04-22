from google.protobuf import any_pb2 as _any_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

CODE_ALREADY_EXISTS: Code
CODE_CANCELLED: Code
CODE_DEADLINE_EXCEEDED: Code
CODE_DO_NOT_USE: Code
CODE_FAILED_PRECONDITION: Code
CODE_INTERNAL: Code
CODE_INVALID_ARGUMENT: Code
CODE_NOT_FOUND: Code
CODE_OK: Code
CODE_PERMISSION_DENIED: Code
CODE_RESOURCE_UNAVAILABLE: Code
CODE_SYSTEM: Code
CODE_UNAUTHENTICATED: Code
CODE_UNIMPLEMENTED: Code
CODE_UNKNOWN: Code
DESCRIPTOR: _descriptor.FileDescriptor

class Status(_message.Message):
    __slots__ = ["context", "err_code", "msg"]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    ERR_CODE_FIELD_NUMBER: _ClassVar[int]
    MSG_FIELD_NUMBER: _ClassVar[int]
    context: _any_pb2.Any
    err_code: Code
    msg: str
    def __init__(self, err_code: _Optional[_Union[Code, str]] = ..., msg: _Optional[str] = ..., context: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class Code(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
