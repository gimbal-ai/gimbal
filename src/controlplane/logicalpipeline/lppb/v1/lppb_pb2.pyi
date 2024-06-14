from gogoproto import gogo_pb2 as _gogo_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from src.api.corepb.v1 import model_exec_pb2 as _model_exec_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CreateLogicalPipelineRequest(_message.Message):
    __slots__ = ["name", "org_id", "pipeline", "yaml"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ORG_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    YAML_FIELD_NUMBER: _ClassVar[int]
    name: str
    org_id: _uuid_pb2.UUID
    pipeline: _model_exec_pb2.Pipeline
    yaml: str
    def __init__(self, org_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., name: _Optional[str] = ..., pipeline: _Optional[_Union[_model_exec_pb2.Pipeline, _Mapping]] = ..., yaml: _Optional[str] = ...) -> None: ...

class CreateLogicalPipelineResponse(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetLogicalPipelineRequest(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetLogicalPipelineResponse(_message.Message):
    __slots__ = ["encoded_dag", "logical_pipeline", "pipeline", "pipeline_info", "yaml"]
    ENCODED_DAG_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_PIPELINE_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_INFO_FIELD_NUMBER: _ClassVar[int]
    YAML_FIELD_NUMBER: _ClassVar[int]
    encoded_dag: str
    logical_pipeline: _model_exec_pb2.LogicalPipeline
    pipeline: _model_exec_pb2.Pipeline
    pipeline_info: LogicalPipelineInfo
    yaml: str
    def __init__(self, pipeline: _Optional[_Union[_model_exec_pb2.Pipeline, _Mapping]] = ..., pipeline_info: _Optional[_Union[LogicalPipelineInfo, _Mapping]] = ..., yaml: _Optional[str] = ..., logical_pipeline: _Optional[_Union[_model_exec_pb2.LogicalPipeline, _Mapping]] = ..., encoded_dag: _Optional[str] = ...) -> None: ...

class ListLogicalPipelinesRequest(_message.Message):
    __slots__ = ["org_id"]
    ORG_ID_FIELD_NUMBER: _ClassVar[int]
    org_id: _uuid_pb2.UUID
    def __init__(self, org_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class ListLogicalPipelinesResponse(_message.Message):
    __slots__ = ["pipelines"]
    PIPELINES_FIELD_NUMBER: _ClassVar[int]
    pipelines: _containers.RepeatedCompositeFieldContainer[LogicalPipelineInfo]
    def __init__(self, pipelines: _Optional[_Iterable[_Union[LogicalPipelineInfo, _Mapping]]] = ...) -> None: ...

class LogicalPipelineInfo(_message.Message):
    __slots__ = ["id", "name"]
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    name: str
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., name: _Optional[str] = ...) -> None: ...
