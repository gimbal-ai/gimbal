from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PingClientStreamRequest(_message.Message):
    __slots__ = ["req"]
    REQ_FIELD_NUMBER: _ClassVar[int]
    req: str
    def __init__(self, req: _Optional[str] = ...) -> None: ...

class PingClientStreamResponse(_message.Message):
    __slots__ = ["reply"]
    REPLY_FIELD_NUMBER: _ClassVar[int]
    reply: str
    def __init__(self, reply: _Optional[str] = ...) -> None: ...

class PingRequest(_message.Message):
    __slots__ = ["req"]
    REQ_FIELD_NUMBER: _ClassVar[int]
    req: str
    def __init__(self, req: _Optional[str] = ...) -> None: ...

class PingResponse(_message.Message):
    __slots__ = ["reply"]
    REPLY_FIELD_NUMBER: _ClassVar[int]
    reply: str
    def __init__(self, reply: _Optional[str] = ...) -> None: ...

class PingServerStreamRequest(_message.Message):
    __slots__ = ["req"]
    REQ_FIELD_NUMBER: _ClassVar[int]
    req: str
    def __init__(self, req: _Optional[str] = ...) -> None: ...

class PingServerStreamResponse(_message.Message):
    __slots__ = ["reply"]
    REPLY_FIELD_NUMBER: _ClassVar[int]
    reply: str
    def __init__(self, reply: _Optional[str] = ...) -> None: ...
