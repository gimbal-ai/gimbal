from gogoproto import gogo_pb2 as _gogo_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from src.common.typespb import status_pb2 as _status_pb2
from google.protobuf import any_pb2 as _any_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from opentelemetry.proto.metrics.v1 import metrics_pb2 as _metrics_pb2
from src.api.corepb.v1 import model_exec_pb2 as _model_exec_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

CP_EDGE_TOPIC_EXEC: CPEdgeTopic
CP_EDGE_TOPIC_FILE_TRANSFER: CPEdgeTopic
CP_EDGE_TOPIC_INFO: CPEdgeTopic
CP_EDGE_TOPIC_METRICS: CPEdgeTopic
CP_EDGE_TOPIC_STATUS: CPEdgeTopic
CP_EDGE_TOPIC_UNKNOWN: CPEdgeTopic
CP_EDGE_TOPIC_VIDEO: CPEdgeTopic
DESCRIPTOR: _descriptor.FileDescriptor
EDGE_CP_TOPIC_EXEC: EdgeCPTopic
EDGE_CP_TOPIC_FILE_TRANSFER: EdgeCPTopic
EDGE_CP_TOPIC_INFO: EdgeCPTopic
EDGE_CP_TOPIC_METRICS: EdgeCPTopic
EDGE_CP_TOPIC_STATUS: EdgeCPTopic
EDGE_CP_TOPIC_UNKNOWN: EdgeCPTopic
EDGE_CP_TOPIC_VIDEO: EdgeCPTopic
EXECUTION_GRAPH_STATE_COMPILING: ExecutionGraphState
EXECUTION_GRAPH_STATE_DEPLOYED: ExecutionGraphState
EXECUTION_GRAPH_STATE_DOWNLOADING: ExecutionGraphState
EXECUTION_GRAPH_STATE_FAILED: ExecutionGraphState
EXECUTION_GRAPH_STATE_READY: ExecutionGraphState
EXECUTION_GRAPH_STATE_TERMINATING: ExecutionGraphState
EXECUTION_GRAPH_STATE_UNKNOWN: ExecutionGraphState
EXECUTION_GRAPH_STATE_UPDATE_REQUESTED: ExecutionGraphState

class ApplyExecutionGraph(_message.Message):
    __slots__ = ["logical_pipeline_id", "physical_pipeline_id", "spec"]
    LOGICAL_PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    PHYSICAL_PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    logical_pipeline_id: _uuid_pb2.UUID
    physical_pipeline_id: _uuid_pb2.UUID
    spec: ExecutionGraphSpec
    def __init__(self, physical_pipeline_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., logical_pipeline_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., spec: _Optional[_Union[ExecutionGraphSpec, _Mapping]] = ...) -> None: ...

class CPEdgeMessage(_message.Message):
    __slots__ = ["metadata", "msg"]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    MSG_FIELD_NUMBER: _ClassVar[int]
    metadata: CPEdgeMetadata
    msg: _any_pb2.Any
    def __init__(self, metadata: _Optional[_Union[CPEdgeMetadata, _Mapping]] = ..., msg: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class CPEdgeMetadata(_message.Message):
    __slots__ = ["device_id", "recv_timestamp", "topic"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    RECV_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    recv_timestamp: _timestamp_pb2.Timestamp
    topic: CPEdgeTopic
    def __init__(self, topic: _Optional[_Union[CPEdgeTopic, str]] = ..., device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., recv_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class CPRunModel(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class CPRunModelAck(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class DeleteExecutionGraph(_message.Message):
    __slots__ = ["physical_pipeline_id"]
    PHYSICAL_PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    physical_pipeline_id: _uuid_pb2.UUID
    def __init__(self, physical_pipeline_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class DeviceCapabilities(_message.Message):
    __slots__ = ["cameras", "model_runtimes"]
    class CameraInfo(_message.Message):
        __slots__ = ["camera_id", "driver"]
        class CameraDriver(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        CAMERA_DRIVER_ARGUS: DeviceCapabilities.CameraInfo.CameraDriver
        CAMERA_DRIVER_UNKNOWN: DeviceCapabilities.CameraInfo.CameraDriver
        CAMERA_DRIVER_V4L2: DeviceCapabilities.CameraInfo.CameraDriver
        CAMERA_ID_FIELD_NUMBER: _ClassVar[int]
        DRIVER_FIELD_NUMBER: _ClassVar[int]
        camera_id: str
        driver: DeviceCapabilities.CameraInfo.CameraDriver
        def __init__(self, driver: _Optional[_Union[DeviceCapabilities.CameraInfo.CameraDriver, str]] = ..., camera_id: _Optional[str] = ...) -> None: ...
    class ModelRuntimeInfo(_message.Message):
        __slots__ = ["type"]
        class ModelRuntimeType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        MODEL_RUNTIME_TYPE_OPENVINO: DeviceCapabilities.ModelRuntimeInfo.ModelRuntimeType
        MODEL_RUNTIME_TYPE_TENSORRT: DeviceCapabilities.ModelRuntimeInfo.ModelRuntimeType
        MODEL_RUNTIME_TYPE_UNKNOWN: DeviceCapabilities.ModelRuntimeInfo.ModelRuntimeType
        TYPE_FIELD_NUMBER: _ClassVar[int]
        type: DeviceCapabilities.ModelRuntimeInfo.ModelRuntimeType
        def __init__(self, type: _Optional[_Union[DeviceCapabilities.ModelRuntimeInfo.ModelRuntimeType, str]] = ...) -> None: ...
    CAMERAS_FIELD_NUMBER: _ClassVar[int]
    MODEL_RUNTIMES_FIELD_NUMBER: _ClassVar[int]
    cameras: _containers.RepeatedCompositeFieldContainer[DeviceCapabilities.CameraInfo]
    model_runtimes: _containers.RepeatedCompositeFieldContainer[DeviceCapabilities.ModelRuntimeInfo]
    def __init__(self, model_runtimes: _Optional[_Iterable[_Union[DeviceCapabilities.ModelRuntimeInfo, _Mapping]]] = ..., cameras: _Optional[_Iterable[_Union[DeviceCapabilities.CameraInfo, _Mapping]]] = ...) -> None: ...

class EdgeCPMessage(_message.Message):
    __slots__ = ["metadata", "msg"]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    MSG_FIELD_NUMBER: _ClassVar[int]
    metadata: EdgeCPMetadata
    msg: _any_pb2.Any
    def __init__(self, metadata: _Optional[_Union[EdgeCPMetadata, _Mapping]] = ..., msg: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class EdgeCPMetadata(_message.Message):
    __slots__ = ["device_id", "recv_timestamp", "topic"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    RECV_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    recv_timestamp: _timestamp_pb2.Timestamp
    topic: EdgeCPTopic
    def __init__(self, topic: _Optional[_Union[EdgeCPTopic, str]] = ..., device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., recv_timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class EdgeHeartbeat(_message.Message):
    __slots__ = ["seq_id"]
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    seq_id: int
    def __init__(self, seq_id: _Optional[int] = ...) -> None: ...

class EdgeHeartbeatAck(_message.Message):
    __slots__ = ["seq_id"]
    SEQ_ID_FIELD_NUMBER: _ClassVar[int]
    seq_id: int
    def __init__(self, seq_id: _Optional[int] = ...) -> None: ...

class EdgeOTelMetrics(_message.Message):
    __slots__ = ["resource_metrics"]
    RESOURCE_METRICS_FIELD_NUMBER: _ClassVar[int]
    resource_metrics: _metrics_pb2.ResourceMetrics
    def __init__(self, resource_metrics: _Optional[_Union[_metrics_pb2.ResourceMetrics, _Mapping]] = ...) -> None: ...

class ExecutionGraphSpec(_message.Message):
    __slots__ = ["graph", "state", "version"]
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    graph: _model_exec_pb2.ExecutionSpec
    state: ExecutionGraphState
    version: int
    def __init__(self, graph: _Optional[_Union[_model_exec_pb2.ExecutionSpec, _Mapping]] = ..., state: _Optional[_Union[ExecutionGraphState, str]] = ..., version: _Optional[int] = ...) -> None: ...

class ExecutionGraphStatus(_message.Message):
    __slots__ = ["reason", "state", "version"]
    REASON_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    reason: str
    state: ExecutionGraphState
    version: int
    def __init__(self, state: _Optional[_Union[ExecutionGraphState, str]] = ..., reason: _Optional[str] = ..., version: _Optional[int] = ...) -> None: ...

class ExecutionGraphStatusUpdate(_message.Message):
    __slots__ = ["physical_pipeline_id", "status"]
    PHYSICAL_PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    physical_pipeline_id: _uuid_pb2.UUID
    status: ExecutionGraphStatus
    def __init__(self, physical_pipeline_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., status: _Optional[_Union[ExecutionGraphStatus, _Mapping]] = ...) -> None: ...

class FileTransferRequest(_message.Message):
    __slots__ = ["chunk_start_bytes", "file_id", "num_bytes"]
    CHUNK_START_BYTES_FIELD_NUMBER: _ClassVar[int]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    NUM_BYTES_FIELD_NUMBER: _ClassVar[int]
    chunk_start_bytes: int
    file_id: _uuid_pb2.UUID
    num_bytes: int
    def __init__(self, file_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., chunk_start_bytes: _Optional[int] = ..., num_bytes: _Optional[int] = ...) -> None: ...

class FileTransferResponse(_message.Message):
    __slots__ = ["chunk", "file_id", "status"]
    class FileChunk(_message.Message):
        __slots__ = ["payload", "start_bytes"]
        PAYLOAD_FIELD_NUMBER: _ClassVar[int]
        START_BYTES_FIELD_NUMBER: _ClassVar[int]
        payload: bytes
        start_bytes: int
        def __init__(self, start_bytes: _Optional[int] = ..., payload: _Optional[bytes] = ...) -> None: ...
    CHUNK_FIELD_NUMBER: _ClassVar[int]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    chunk: FileTransferResponse.FileChunk
    file_id: _uuid_pb2.UUID
    status: _status_pb2.Status
    def __init__(self, status: _Optional[_Union[_status_pb2.Status, _Mapping]] = ..., chunk: _Optional[_Union[FileTransferResponse.FileChunk, _Mapping]] = ..., file_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class VideoStreamKeepAlive(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class VideoStreamStart(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class VideoStreamStop(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ExecutionGraphState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class EdgeCPTopic(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class CPEdgeTopic(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
