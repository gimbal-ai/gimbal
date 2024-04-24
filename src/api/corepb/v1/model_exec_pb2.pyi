from gogoproto import gogo_pb2 as _gogo_pb2
from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

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

class Pipeline(_message.Message):
    __slots__ = ["nodes"]
    NODES_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[PipelineNode]
    def __init__(self, nodes: _Optional[_Iterable[_Union[PipelineNode, _Mapping]]] = ...) -> None: ...

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