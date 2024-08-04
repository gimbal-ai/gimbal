from qdrant import collections_service_pb2 as _collections_service_pb2
from qdrant import points_service_pb2 as _points_service_pb2
from qdrant import snapshots_service_pb2 as _snapshots_service_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class HealthCheckReply(_message.Message):
    __slots__ = ["commit", "title", "version"]
    COMMIT_FIELD_NUMBER: _ClassVar[int]
    TITLE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    commit: str
    title: str
    version: str
    def __init__(self, title: _Optional[str] = ..., version: _Optional[str] = ..., commit: _Optional[str] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...
