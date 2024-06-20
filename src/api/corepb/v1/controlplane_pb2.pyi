from gogoproto import gogo_pb2 as _gogo_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

CP_TOPIC_DEVICE_CONNECTED: CPTopic
CP_TOPIC_DEVICE_DISCONNECTED: CPTopic
CP_TOPIC_DEVICE_UPDATE: CPTopic
CP_TOPIC_PHYSICAL_PIPELINE_RECONCILIATION: CPTopic
CP_TOPIC_PIPELINE_DEPLOYMENT_RECONCILIATION: CPTopic
CP_TOPIC_UNKNOWN: CPTopic
DESCRIPTOR: _descriptor.FileDescriptor

class CPMessage(_message.Message):
    __slots__ = ["metadata", "msg"]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    MSG_FIELD_NUMBER: _ClassVar[int]
    metadata: CPMetadata
    msg: _any_pb2.Any
    def __init__(self, metadata: _Optional[_Union[CPMetadata, _Mapping]] = ..., msg: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class CPMetadata(_message.Message):
    __slots__ = ["entity_id", "recv_timestamp", "topic"]
    ENTITY_ID_FIELD_NUMBER: _ClassVar[int]
    RECV_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    entity_id: _uuid_pb2.UUID
    recv_timestamp: _timestamp_pb2.Timestamp
    topic: CPTopic
    def __init__(self, topic: _Optional[_Union[CPTopic, str]] = ..., entity_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., recv_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class DeviceConnected(_message.Message):
    __slots__ = ["device_id"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class DeviceDisconnected(_message.Message):
    __slots__ = ["device_id"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class DeviceUpdate(_message.Message):
    __slots__ = ["device_id"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class PhysicalPipelineReconciliation(_message.Message):
    __slots__ = ["device_id", "force_apply", "physical_pipeline_id"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    FORCE_APPLY_FIELD_NUMBER: _ClassVar[int]
    PHYSICAL_PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    force_apply: bool
    physical_pipeline_id: _uuid_pb2.UUID
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., physical_pipeline_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., force_apply: bool = ...) -> None: ...

class PipelineDeploymentReconciliation(_message.Message):
    __slots__ = ["fleet_id", "pipeline_deployment_id"]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    fleet_id: _uuid_pb2.UUID
    pipeline_deployment_id: _uuid_pb2.UUID
    def __init__(self, pipeline_deployment_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., fleet_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class CPTopic(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
