from gogoproto import gogo_pb2 as _gogo_pb2
from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor
LOGICAL_PIPELINE_NODE_KIND_CAMERA_SOURCE: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_CLASSIFY: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_DETECT: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_DETECTIONS_METRICS_SINK: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_FOR_EACH_ROI: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_FRAME_METRICS_SINK: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_INPUT: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_LATENCY_METRICS_SINK: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_OUTPUT: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_UNKNOWN: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_VIDEO_STREAM_SINK: LogicalPipelineNodeKind
PIPELINE_STATE_FAILED: PipelineState
PIPELINE_STATE_PENDING: PipelineState
PIPELINE_STATE_READY: PipelineState
PIPELINE_STATE_RUNNING: PipelineState
PIPELINE_STATE_TERMINATED: PipelineState
PIPELINE_STATE_TERMINATING: PipelineState
PIPELINE_STATE_UNKNOWN: PipelineState

class ExecutionSpec(_message.Message):
    __slots__ = ["graph", "model_spec"]
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    MODEL_SPEC_FIELD_NUMBER: _ClassVar[int]
    graph: _calculator_pb2.CalculatorGraphConfig
    model_spec: _containers.RepeatedCompositeFieldContainer[ModelSpec]
    def __init__(self, graph: _Optional[_Union[_calculator_pb2.CalculatorGraphConfig, _Mapping]] = ..., model_spec: _Optional[_Iterable[_Union[ModelSpec, _Mapping]]] = ...) -> None: ...

class FileResource(_message.Message):
    __slots__ = ["file_id", "sha256_hash", "size_bytes"]
    FILE_ID_FIELD_NUMBER: _ClassVar[int]
    SHA256_HASH_FIELD_NUMBER: _ClassVar[int]
    SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    file_id: _uuid_pb2.UUID
    sha256_hash: str
    size_bytes: int
    def __init__(self, file_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., size_bytes: _Optional[int] = ..., sha256_hash: _Optional[str] = ...) -> None: ...

class GlobalParam(_message.Message):
    __slots__ = ["bool_value", "double_value", "int64_value", "name", "string_value"]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT64_VALUE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    bool_value: bool
    double_value: float
    int64_value: int
    name: str
    string_value: str
    def __init__(self, name: _Optional[str] = ..., string_value: _Optional[str] = ..., int64_value: _Optional[int] = ..., double_value: _Optional[float] = ..., bool_value: bool = ...) -> None: ...

class Lambda(_message.Message):
    __slots__ = ["nodes"]
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    def __init__(self, nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ...) -> None: ...

class LogicalPipeline(_message.Message):
    __slots__ = ["global_params", "nodes"]
    GLOBAL_PARAMS_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    global_params: _containers.RepeatedCompositeFieldContainer[GlobalParam]
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    def __init__(self, global_params: _Optional[_Iterable[_Union[GlobalParam, _Mapping]]] = ..., nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ...) -> None: ...

class ModelSpec(_message.Message):
    __slots__ = ["name", "onnx_blob_key", "onnx_file", "openvino_spec", "runtime", "tensorrt_spec"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ONNX_BLOB_KEY_FIELD_NUMBER: _ClassVar[int]
    ONNX_FILE_FIELD_NUMBER: _ClassVar[int]
    OPENVINO_SPEC_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    TENSORRT_SPEC_FIELD_NUMBER: _ClassVar[int]
    name: str
    onnx_blob_key: str
    onnx_file: FileResource
    openvino_spec: OpenVINOModelSpec
    runtime: str
    tensorrt_spec: TensorRTModelSpec
    def __init__(self, name: _Optional[str] = ..., onnx_blob_key: _Optional[str] = ..., onnx_file: _Optional[_Union[FileResource, _Mapping]] = ..., runtime: _Optional[str] = ..., tensorrt_spec: _Optional[_Union[TensorRTModelSpec, _Mapping]] = ..., openvino_spec: _Optional[_Union[OpenVINOModelSpec, _Mapping]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ["init_args", "inputs", "kind", "name", "outputs"]
    INIT_ARGS_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    init_args: _containers.RepeatedCompositeFieldContainer[NodeInitArgs]
    inputs: _containers.RepeatedCompositeFieldContainer[NodeInput]
    kind: LogicalPipelineNodeKind
    name: str
    outputs: _containers.RepeatedCompositeFieldContainer[NodeOutput]
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[_Union[LogicalPipelineNodeKind, str]] = ..., inputs: _Optional[_Iterable[_Union[NodeInput, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[NodeOutput, _Mapping]]] = ..., init_args: _Optional[_Iterable[_Union[NodeInitArgs, _Mapping]]] = ...) -> None: ...

class NodeInitArgs(_message.Message):
    __slots__ = ["bool_value", "double_value", "int64_value", "lambda_value", "name", "string_value"]
    BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
    INT64_VALUE_FIELD_NUMBER: _ClassVar[int]
    LAMBDA_VALUE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    bool_value: bool
    double_value: float
    int64_value: int
    lambda_value: Lambda
    name: str
    string_value: str
    def __init__(self, name: _Optional[str] = ..., string_value: _Optional[str] = ..., int64_value: _Optional[int] = ..., double_value: _Optional[float] = ..., bool_value: bool = ..., lambda_value: _Optional[_Union[Lambda, _Mapping]] = ...) -> None: ...

class NodeInput(_message.Message):
    __slots__ = ["model_value", "name", "node_output_value", "param_value"]
    class ModelInput(_message.Message):
        __slots__ = ["model_name"]
        MODEL_NAME_FIELD_NUMBER: _ClassVar[int]
        model_name: str
        def __init__(self, model_name: _Optional[str] = ...) -> None: ...
    class NodeOutputRef(_message.Message):
        __slots__ = ["name", "node_name"]
        NAME_FIELD_NUMBER: _ClassVar[int]
        NODE_NAME_FIELD_NUMBER: _ClassVar[int]
        name: str
        node_name: int
        def __init__(self, node_name: _Optional[int] = ..., name: _Optional[str] = ...) -> None: ...
    class ParamInput(_message.Message):
        __slots__ = ["param_name"]
        PARAM_NAME_FIELD_NUMBER: _ClassVar[int]
        param_name: str
        def __init__(self, param_name: _Optional[str] = ...) -> None: ...
    MODEL_VALUE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NODE_OUTPUT_VALUE_FIELD_NUMBER: _ClassVar[int]
    PARAM_VALUE_FIELD_NUMBER: _ClassVar[int]
    model_value: NodeInput.ModelInput
    name: str
    node_output_value: NodeInput.NodeOutputRef
    param_value: NodeInput.ParamInput
    def __init__(self, name: _Optional[str] = ..., param_value: _Optional[_Union[NodeInput.ParamInput, _Mapping]] = ..., model_value: _Optional[_Union[NodeInput.ModelInput, _Mapping]] = ..., node_output_value: _Optional[_Union[NodeInput.NodeOutputRef, _Mapping]] = ...) -> None: ...

class NodeOutput(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class OpenVINOModelSpec(_message.Message):
    __slots__ = ["input_shape"]
    class TensorShape(_message.Message):
        __slots__ = ["dim"]
        DIM_FIELD_NUMBER: _ClassVar[int]
        dim: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, dim: _Optional[_Iterable[int]] = ...) -> None: ...
    INPUT_SHAPE_FIELD_NUMBER: _ClassVar[int]
    input_shape: _containers.RepeatedCompositeFieldContainer[OpenVINOModelSpec.TensorShape]
    def __init__(self, input_shape: _Optional[_Iterable[_Union[OpenVINOModelSpec.TensorShape, _Mapping]]] = ...) -> None: ...

class PhysicalPipeline(_message.Message):
    __slots__ = ["created_at", "device_id", "id", "pipeline_deployment_id", "spec", "status", "updated_at", "version"]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    created_at: _timestamp_pb2.Timestamp
    device_id: _uuid_pb2.UUID
    id: _uuid_pb2.UUID
    pipeline_deployment_id: _uuid_pb2.UUID
    spec: PhysicalPipelineSpec
    status: PhysicalPipelineStatus
    updated_at: _timestamp_pb2.Timestamp
    version: int
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., pipeline_deployment_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., version: _Optional[int] = ..., spec: _Optional[_Union[PhysicalPipelineSpec, _Mapping]] = ..., status: _Optional[_Union[PhysicalPipelineStatus, _Mapping]] = ...) -> None: ...

class PhysicalPipelineSpec(_message.Message):
    __slots__ = ["state"]
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: PipelineState
    def __init__(self, state: _Optional[_Union[PipelineState, str]] = ...) -> None: ...

class PhysicalPipelineStatus(_message.Message):
    __slots__ = ["state"]
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: PipelineState
    def __init__(self, state: _Optional[_Union[PipelineState, str]] = ...) -> None: ...

class Pipeline(_message.Message):
    __slots__ = ["nodes"]
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[PipelineNode]
    def __init__(self, nodes: _Optional[_Iterable[_Union[PipelineNode, _Mapping]]] = ...) -> None: ...

class PipelineDeployment(_message.Message):
    __slots__ = ["created_at", "fleet_id", "id", "logical_pipeline_id", "spec", "status", "updated_at", "version"]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    created_at: _timestamp_pb2.Timestamp
    fleet_id: _uuid_pb2.UUID
    id: _uuid_pb2.UUID
    logical_pipeline_id: _uuid_pb2.UUID
    spec: PipelineDeploymentSpec
    status: PipelineDeploymentStatus
    updated_at: _timestamp_pb2.Timestamp
    version: int
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., logical_pipeline_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., fleet_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., version: _Optional[int] = ..., spec: _Optional[_Union[PipelineDeploymentSpec, _Mapping]] = ..., status: _Optional[_Union[PipelineDeploymentStatus, _Mapping]] = ...) -> None: ...

class PipelineDeploymentSpec(_message.Message):
    __slots__ = ["state"]
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: PipelineState
    def __init__(self, state: _Optional[_Union[PipelineState, str]] = ...) -> None: ...

class PipelineDeploymentStatus(_message.Message):
    __slots__ = ["state"]
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: PipelineState
    def __init__(self, state: _Optional[_Union[PipelineState, str]] = ...) -> None: ...

class PipelineNode(_message.Message):
    __slots__ = ["attr", "id", "inputs", "outputs", "type"]
    class AttrEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ATTR_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    attr: _containers.ScalarMap[str, str]
    id: int
    inputs: _containers.RepeatedCompositeFieldContainer[Port]
    outputs: _containers.RepeatedCompositeFieldContainer[Port]
    type: str
    def __init__(self, id: _Optional[int] = ..., type: _Optional[str] = ..., inputs: _Optional[_Iterable[_Union[Port, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[Port, _Mapping]]] = ..., attr: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Port(_message.Message):
    __slots__ = ["name", "net"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NET_FIELD_NUMBER: _ClassVar[int]
    name: str
    net: str
    def __init__(self, name: _Optional[str] = ..., net: _Optional[str] = ...) -> None: ...

class TensorRTMemPoolLimits(_message.Message):
    __slots__ = ["workspace"]
    WORKSPACE_FIELD_NUMBER: _ClassVar[int]
    workspace: int
    def __init__(self, workspace: _Optional[int] = ...) -> None: ...

class TensorRTModelSpec(_message.Message):
    __slots__ = ["engine_blob_key", "mem_pool_limits", "optimization_profile"]
    ENGINE_BLOB_KEY_FIELD_NUMBER: _ClassVar[int]
    MEM_POOL_LIMITS_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZATION_PROFILE_FIELD_NUMBER: _ClassVar[int]
    engine_blob_key: str
    mem_pool_limits: TensorRTMemPoolLimits
    optimization_profile: _containers.RepeatedCompositeFieldContainer[TensorRTOptimizationProfile]
    def __init__(self, optimization_profile: _Optional[_Iterable[_Union[TensorRTOptimizationProfile, _Mapping]]] = ..., engine_blob_key: _Optional[str] = ..., mem_pool_limits: _Optional[_Union[TensorRTMemPoolLimits, _Mapping]] = ...) -> None: ...

class TensorRTOptimizationProfile(_message.Message):
    __slots__ = ["tensor_shape_range"]
    TENSOR_SHAPE_RANGE_FIELD_NUMBER: _ClassVar[int]
    tensor_shape_range: _containers.RepeatedCompositeFieldContainer[TensorRTTensorShapeRange]
    def __init__(self, tensor_shape_range: _Optional[_Iterable[_Union[TensorRTTensorShapeRange, _Mapping]]] = ...) -> None: ...

class TensorRTTensorShapeRange(_message.Message):
    __slots__ = ["dim", "tensor_name"]
    DIM_FIELD_NUMBER: _ClassVar[int]
    TENSOR_NAME_FIELD_NUMBER: _ClassVar[int]
    dim: _containers.RepeatedScalarFieldContainer[int]
    tensor_name: str
    def __init__(self, tensor_name: _Optional[str] = ..., dim: _Optional[_Iterable[int]] = ...) -> None: ...

class LogicalPipelineNodeKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class PipelineState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
