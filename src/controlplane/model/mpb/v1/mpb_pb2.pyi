from gogoproto import gogo_pb2 as _gogo_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from src.api.corepb.v1 import model_exec_pb2 as _model_exec_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CreateModelRequest(_message.Message):
    __slots__ = ["model_info", "name", "org_id"]
    MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ORG_ID_FIELD_NUMBER: _ClassVar[int]
    model_info: _model_exec_pb2.ModelInfo
    name: str
    org_id: _uuid_pb2.UUID
    def __init__(self, org_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., name: _Optional[str] = ..., model_info: _Optional[_Union[_model_exec_pb2.ModelInfo, _Mapping]] = ...) -> None: ...

class CreateModelResponse(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetModelRequest(_message.Message):
    __slots__ = ["id", "name", "org_id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ORG_ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    name: str
    org_id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., name: _Optional[str] = ..., org_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetModelResponse(_message.Message):
    __slots__ = ["model_info"]
    MODEL_INFO_FIELD_NUMBER: _ClassVar[int]
    model_info: _model_exec_pb2.ModelInfo
    def __init__(self, model_info: _Optional[_Union[_model_exec_pb2.ModelInfo, _Mapping]] = ...) -> None: ...
