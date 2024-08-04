from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CreateFullSnapshotRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class CreateSnapshotRequest(_message.Message):
    __slots__ = ["collection_name"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    def __init__(self, collection_name: _Optional[str] = ...) -> None: ...

class CreateSnapshotResponse(_message.Message):
    __slots__ = ["snapshot_description", "time"]
    SNAPSHOT_DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    snapshot_description: SnapshotDescription
    time: float
    def __init__(self, snapshot_description: _Optional[_Union[SnapshotDescription, _Mapping]] = ..., time: _Optional[float] = ...) -> None: ...

class DeleteFullSnapshotRequest(_message.Message):
    __slots__ = ["snapshot_name"]
    SNAPSHOT_NAME_FIELD_NUMBER: _ClassVar[int]
    snapshot_name: str
    def __init__(self, snapshot_name: _Optional[str] = ...) -> None: ...

class DeleteSnapshotRequest(_message.Message):
    __slots__ = ["collection_name", "snapshot_name"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    SNAPSHOT_NAME_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    snapshot_name: str
    def __init__(self, collection_name: _Optional[str] = ..., snapshot_name: _Optional[str] = ...) -> None: ...

class DeleteSnapshotResponse(_message.Message):
    __slots__ = ["time"]
    TIME_FIELD_NUMBER: _ClassVar[int]
    time: float
    def __init__(self, time: _Optional[float] = ...) -> None: ...

class ListFullSnapshotsRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ListSnapshotsRequest(_message.Message):
    __slots__ = ["collection_name"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    def __init__(self, collection_name: _Optional[str] = ...) -> None: ...

class ListSnapshotsResponse(_message.Message):
    __slots__ = ["snapshot_descriptions", "time"]
    SNAPSHOT_DESCRIPTIONS_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    snapshot_descriptions: _containers.RepeatedCompositeFieldContainer[SnapshotDescription]
    time: float
    def __init__(self, snapshot_descriptions: _Optional[_Iterable[_Union[SnapshotDescription, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class SnapshotDescription(_message.Message):
    __slots__ = ["checksum", "creation_time", "name", "size"]
    CHECKSUM_FIELD_NUMBER: _ClassVar[int]
    CREATION_TIME_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    checksum: str
    creation_time: _timestamp_pb2.Timestamp
    name: str
    size: int
    def __init__(self, name: _Optional[str] = ..., creation_time: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., size: _Optional[int] = ..., checksum: _Optional[str] = ...) -> None: ...
