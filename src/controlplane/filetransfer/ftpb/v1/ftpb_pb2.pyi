from gogoproto import gogo_pb2 as _gogo_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor
FILE_STATUS_CREATED: FileStatus
FILE_STATUS_DELETED: FileStatus
FILE_STATUS_READY: FileStatus
FILE_STATUS_UNKNOWN: FileStatus

class CreateFileInfoRequest(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class CreateFileInfoResponse(_message.Message):
    __slots__ = ["info"]
    INFO_FIELD_NUMBER: _ClassVar[int]
    info: FileInfo
    def __init__(self, info: _Optional[_Union[FileInfo, _Mapping]] = ...) -> None: ...

class DeleteFileRequest(_message.Message):
    __slots__ = ["file_id", "purge"]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    PURGE_FIELD_NUMBER: _ClassVar[int]
    file_id: _uuid_pb2.UUID
    purge: bool
    def __init__(self, file_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., purge: bool = ...) -> None: ...

class DeleteFileResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class DownloadFileRequest(_message.Message):
    __slots__ = ["file_id"]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    file_id: _uuid_pb2.UUID
    def __init__(self, file_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class DownloadFileResponse(_message.Message):
    __slots__ = ["chunk"]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    chunk: bytes
    def __init__(self, chunk: _Optional[bytes] = ...) -> None: ...

class FileInfo(_message.Message):
    __slots__ = ["file_id", "name", "sha256sum", "size_bytes", "status"]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SHA256SUM_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    file_id: _uuid_pb2.UUID
    name: str
    sha256sum: str
    size_bytes: int
    status: FileStatus
    def __init__(self, file_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., status: _Optional[_Union[FileStatus, str]] = ..., name: _Optional[str] = ..., size_bytes: _Optional[int] = ..., sha256sum: _Optional[str] = ...) -> None: ...

class GetFileInfoByNameRequest(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class GetFileInfoByNameResponse(_message.Message):
    __slots__ = ["info"]
    INFO_FIELD_NUMBER: _ClassVar[int]
    info: FileInfo
    def __init__(self, info: _Optional[_Union[FileInfo, _Mapping]] = ...) -> None: ...

class GetFileInfoRequest(_message.Message):
    __slots__ = ["file_id"]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    file_id: _uuid_pb2.UUID
    def __init__(self, file_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetFileInfoResponse(_message.Message):
    __slots__ = ["info"]
    INFO_FIELD_NUMBER: _ClassVar[int]
    info: FileInfo
    def __init__(self, info: _Optional[_Union[FileInfo, _Mapping]] = ...) -> None: ...

class UploadFileRequest(_message.Message):
    __slots__ = ["chunk", "file_id", "sha256sum"]
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    SHA256SUM_FIELD_NUMBER: _ClassVar[int]
    chunk: bytes
    file_id: _uuid_pb2.UUID
    sha256sum: str
    def __init__(self, file_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., sha256sum: _Optional[str] = ..., chunk: _Optional[bytes] = ...) -> None: ...

class UploadFileResponse(_message.Message):
    __slots__ = ["file_id", "size_bytes"]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    file_id: _uuid_pb2.UUID
    size_bytes: int
    def __init__(self, file_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., size_bytes: _Optional[int] = ...) -> None: ...

class FileStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
