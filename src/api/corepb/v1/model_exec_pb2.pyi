from gogoproto import gogo_pb2 as _gogo_pb2
from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from google.protobuf import wrappers_pb2 as _wrappers_pb2
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
LOGICAL_PIPELINE_NODE_KIND_LATENCY_METRICS_SINK: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_MULTI_PURPOSE_MODEL: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_REGRESS: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_SEGMENT: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_TEXT_STREAM_SINK: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_TEXT_STREAM_SOURCE: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_TRACK: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_UNKNOWN: LogicalPipelineNodeKind
LOGICAL_PIPELINE_NODE_KIND_VIDEO_STREAM_SINK: LogicalPipelineNodeKind
PIPELINE_STATE_FAILED: PipelineState
PIPELINE_STATE_PENDING: PipelineState
PIPELINE_STATE_READY: PipelineState
PIPELINE_STATE_RUNNING: PipelineState
PIPELINE_STATE_TERMINATED: PipelineState
PIPELINE_STATE_TERMINATING: PipelineState
PIPELINE_STATE_UNKNOWN: PipelineState

class BoundingBoxInfo(_message.Message):
    __slots__ = ["box_format", "box_normalized"]
    class BoundingBoxFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BOUNDING_BOX_FORMAT_CXCYWH: BoundingBoxInfo.BoundingBoxFormat
    BOUNDING_BOX_FORMAT_UNKNOWN: BoundingBoxInfo.BoundingBoxFormat
    BOUNDING_BOX_FORMAT_XYXY: BoundingBoxInfo.BoundingBoxFormat
    BOUNDING_BOX_FORMAT_YXYX: BoundingBoxInfo.BoundingBoxFormat
    BOX_FORMAT_FIELD_NUMBER: _ClassVar[int]
    BOX_NORMALIZED_FIELD_NUMBER: _ClassVar[int]
    box_format: BoundingBoxInfo.BoundingBoxFormat
    box_normalized: bool
    def __init__(self, box_format: _Optional[_Union[BoundingBoxInfo.BoundingBoxFormat, str]] = ..., box_normalized: bool = ...) -> None: ...

class DimensionSemantics(_message.Message):
    __slots__ = ["detection_candidates_params", "detection_output_params", "image_channel_params", "kind", "regression_params", "segmentation_mask_params"]
    class DimensionSemanticsKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class DetectionCandidatesParams(_message.Message):
        __slots__ = ["is_nms_boxes"]
        IS_NMS_BOXES_FIELD_NUMBER: _ClassVar[int]
        is_nms_boxes: bool
        def __init__(self, is_nms_boxes: bool = ...) -> None: ...
    class DetectionOutputParams(_message.Message):
        __slots__ = ["box_confidence_index", "box_coordinate_range", "box_format", "class_index", "scores_range"]
        class IndexRange(_message.Message):
            __slots__ = ["size", "start"]
            SIZE_FIELD_NUMBER: _ClassVar[int]
            START_FIELD_NUMBER: _ClassVar[int]
            size: int
            start: int
            def __init__(self, start: _Optional[int] = ..., size: _Optional[int] = ...) -> None: ...
        BOX_CONFIDENCE_INDEX_FIELD_NUMBER: _ClassVar[int]
        BOX_COORDINATE_RANGE_FIELD_NUMBER: _ClassVar[int]
        BOX_FORMAT_FIELD_NUMBER: _ClassVar[int]
        CLASS_INDEX_FIELD_NUMBER: _ClassVar[int]
        SCORES_RANGE_FIELD_NUMBER: _ClassVar[int]
        box_confidence_index: int
        box_coordinate_range: DimensionSemantics.DetectionOutputParams.IndexRange
        box_format: BoundingBoxInfo
        class_index: int
        scores_range: DimensionSemantics.DetectionOutputParams.IndexRange
        def __init__(self, box_coordinate_range: _Optional[_Union[DimensionSemantics.DetectionOutputParams.IndexRange, _Mapping]] = ..., box_format: _Optional[_Union[BoundingBoxInfo, _Mapping]] = ..., box_confidence_index: _Optional[int] = ..., class_index: _Optional[int] = ..., scores_range: _Optional[_Union[DimensionSemantics.DetectionOutputParams.IndexRange, _Mapping]] = ...) -> None: ...
    class ImageChannelParams(_message.Message):
        __slots__ = ["format"]
        class ImageChannelFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        FORMAT_FIELD_NUMBER: _ClassVar[int]
        IMAGE_CHANNEL_FORMAT_BGR: DimensionSemantics.ImageChannelParams.ImageChannelFormat
        IMAGE_CHANNEL_FORMAT_RGB: DimensionSemantics.ImageChannelParams.ImageChannelFormat
        IMAGE_CHANNEL_FORMAT_UNKNOWN: DimensionSemantics.ImageChannelParams.ImageChannelFormat
        format: DimensionSemantics.ImageChannelParams.ImageChannelFormat
        def __init__(self, format: _Optional[_Union[DimensionSemantics.ImageChannelParams.ImageChannelFormat, str]] = ...) -> None: ...
    class RegressionParams(_message.Message):
        __slots__ = ["label", "scale"]
        LABEL_FIELD_NUMBER: _ClassVar[int]
        SCALE_FIELD_NUMBER: _ClassVar[int]
        label: str
        scale: _wrappers_pb2.DoubleValue
        def __init__(self, label: _Optional[str] = ..., scale: _Optional[_Union[_wrappers_pb2.DoubleValue, _Mapping]] = ...) -> None: ...
    class SegmentationMaskParams(_message.Message):
        __slots__ = ["kind"]
        class SegmentationMaskKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        KIND_FIELD_NUMBER: _ClassVar[int]
        SEGMENTATION_MASK_KIND_BOOL: DimensionSemantics.SegmentationMaskParams.SegmentationMaskKind
        SEGMENTATION_MASK_KIND_CLASS_LABEL: DimensionSemantics.SegmentationMaskParams.SegmentationMaskKind
        SEGMENTATION_MASK_KIND_SCORE: DimensionSemantics.SegmentationMaskParams.SegmentationMaskKind
        SEGMENTATION_MASK_KIND_UNKNOWN: DimensionSemantics.SegmentationMaskParams.SegmentationMaskKind
        kind: DimensionSemantics.SegmentationMaskParams.SegmentationMaskKind
        def __init__(self, kind: _Optional[_Union[DimensionSemantics.SegmentationMaskParams.SegmentationMaskKind, str]] = ...) -> None: ...
    DETECTION_CANDIDATES_PARAMS_FIELD_NUMBER: _ClassVar[int]
    DETECTION_OUTPUT_PARAMS_FIELD_NUMBER: _ClassVar[int]
    DIMENSION_SEMANTICS_KIND_BATCH: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_CLASS_LABELS: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_CLASS_SCORES: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_DETECTION_CANDIDATES: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_DETECTION_OUTPUT: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_IGNORE: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_IMAGE_CHANNEL: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_IMAGE_HEIGHT: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_IMAGE_WIDTH: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_REGRESSION_VALUE: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_SEGMENTATION_MASK_CHANNEL: DimensionSemantics.DimensionSemanticsKind
    DIMENSION_SEMANTICS_KIND_UNKNOWN: DimensionSemantics.DimensionSemanticsKind
    IMAGE_CHANNEL_PARAMS_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    REGRESSION_PARAMS_FIELD_NUMBER: _ClassVar[int]
    SEGMENTATION_MASK_PARAMS_FIELD_NUMBER: _ClassVar[int]
    detection_candidates_params: DimensionSemantics.DetectionCandidatesParams
    detection_output_params: DimensionSemantics.DetectionOutputParams
    image_channel_params: DimensionSemantics.ImageChannelParams
    kind: DimensionSemantics.DimensionSemanticsKind
    regression_params: DimensionSemantics.RegressionParams
    segmentation_mask_params: DimensionSemantics.SegmentationMaskParams
    def __init__(self, kind: _Optional[_Union[DimensionSemantics.DimensionSemanticsKind, str]] = ..., image_channel_params: _Optional[_Union[DimensionSemantics.ImageChannelParams, _Mapping]] = ..., detection_candidates_params: _Optional[_Union[DimensionSemantics.DetectionCandidatesParams, _Mapping]] = ..., detection_output_params: _Optional[_Union[DimensionSemantics.DetectionOutputParams, _Mapping]] = ..., segmentation_mask_params: _Optional[_Union[DimensionSemantics.SegmentationMaskParams, _Mapping]] = ..., regression_params: _Optional[_Union[DimensionSemantics.RegressionParams, _Mapping]] = ...) -> None: ...

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

class ImagePreprocessingStep(_message.Message):
    __slots__ = ["conversion_params", "kind", "resize_params", "standardize_params"]
    class ImagePreprocessingKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class ImageConversionParams(_message.Message):
        __slots__ = ["scale"]
        SCALE_FIELD_NUMBER: _ClassVar[int]
        scale: bool
        def __init__(self, scale: bool = ...) -> None: ...
    class ImageResizeParams(_message.Message):
        __slots__ = ["kind"]
        class ImageResizeKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
            __slots__ = []
        IMAGE_RESIZE_KIND_LETTERBOX: ImagePreprocessingStep.ImageResizeParams.ImageResizeKind
        IMAGE_RESIZE_KIND_STRETCH: ImagePreprocessingStep.ImageResizeParams.ImageResizeKind
        IMAGE_RESIZE_KIND_UNKNOWN: ImagePreprocessingStep.ImageResizeParams.ImageResizeKind
        KIND_FIELD_NUMBER: _ClassVar[int]
        kind: ImagePreprocessingStep.ImageResizeParams.ImageResizeKind
        def __init__(self, kind: _Optional[_Union[ImagePreprocessingStep.ImageResizeParams.ImageResizeKind, str]] = ...) -> None: ...
    class ImageStandardizeParams(_message.Message):
        __slots__ = ["means", "stddevs"]
        MEANS_FIELD_NUMBER: _ClassVar[int]
        STDDEVS_FIELD_NUMBER: _ClassVar[int]
        means: _containers.RepeatedScalarFieldContainer[float]
        stddevs: _containers.RepeatedScalarFieldContainer[float]
        def __init__(self, means: _Optional[_Iterable[float]] = ..., stddevs: _Optional[_Iterable[float]] = ...) -> None: ...
    CONVERSION_PARAMS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_PREPROCESSING_KIND_CONVERT_TO_TENSOR: ImagePreprocessingStep.ImagePreprocessingKind
    IMAGE_PREPROCESSING_KIND_RESIZE: ImagePreprocessingStep.ImagePreprocessingKind
    IMAGE_PREPROCESSING_KIND_STANDARDIZE: ImagePreprocessingStep.ImagePreprocessingKind
    IMAGE_PREPROCESSING_KIND_UNKNOWN: ImagePreprocessingStep.ImagePreprocessingKind
    KIND_FIELD_NUMBER: _ClassVar[int]
    RESIZE_PARAMS_FIELD_NUMBER: _ClassVar[int]
    STANDARDIZE_PARAMS_FIELD_NUMBER: _ClassVar[int]
    conversion_params: ImagePreprocessingStep.ImageConversionParams
    kind: ImagePreprocessingStep.ImagePreprocessingKind
    resize_params: ImagePreprocessingStep.ImageResizeParams
    standardize_params: ImagePreprocessingStep.ImageStandardizeParams
    def __init__(self, kind: _Optional[_Union[ImagePreprocessingStep.ImagePreprocessingKind, str]] = ..., conversion_params: _Optional[_Union[ImagePreprocessingStep.ImageConversionParams, _Mapping]] = ..., resize_params: _Optional[_Union[ImagePreprocessingStep.ImageResizeParams, _Mapping]] = ..., standardize_params: _Optional[_Union[ImagePreprocessingStep.ImageStandardizeParams, _Mapping]] = ...) -> None: ...

class LogicalPipeline(_message.Message):
    __slots__ = ["model_ids", "nodes", "params"]
    MODEL_IDS_FIELD_NUMBER: _ClassVar[int]
    NODES_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    model_ids: _containers.RepeatedCompositeFieldContainer[_uuid_pb2.UUID]
    nodes: _containers.RepeatedCompositeFieldContainer[Node]
    params: _containers.RepeatedCompositeFieldContainer[Param]
    def __init__(self, params: _Optional[_Iterable[_Union[Param, _Mapping]]] = ..., nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ..., model_ids: _Optional[_Iterable[_Union[_uuid_pb2.UUID, _Mapping]]] = ...) -> None: ...

class ModelInfo(_message.Message):
    __slots__ = ["bbox_info", "class_labels", "file_assets", "format", "image_preprocessing_steps", "input_tensor_semantics", "kind", "name", "output_tensor_semantics"]
    class ModelKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class ModelStorageFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class FileAssetsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _uuid_pb2.UUID
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...
    BBOX_INFO_FIELD_NUMBER: _ClassVar[int]
    CLASS_LABELS_FIELD_NUMBER: _ClassVar[int]
    FILE_ASSETS_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_PREPROCESSING_STEPS_FIELD_NUMBER: _ClassVar[int]
    INPUT_TENSOR_SEMANTICS_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    MODEL_KIND_ONNX: ModelInfo.ModelKind
    MODEL_KIND_OPENVINO: ModelInfo.ModelKind
    MODEL_KIND_TFLITE: ModelInfo.ModelKind
    MODEL_KIND_TORCH: ModelInfo.ModelKind
    MODEL_KIND_TORCHSCRIPT: ModelInfo.ModelKind
    MODEL_KIND_UNKNOWN: ModelInfo.ModelKind
    MODEL_STORAGE_FORMAT_FLATBUFFER: ModelInfo.ModelStorageFormat
    MODEL_STORAGE_FORMAT_MLIR_BYTECODE: ModelInfo.ModelStorageFormat
    MODEL_STORAGE_FORMAT_MLIR_TEXT: ModelInfo.ModelStorageFormat
    MODEL_STORAGE_FORMAT_OPENVINO: ModelInfo.ModelStorageFormat
    MODEL_STORAGE_FORMAT_PROTOBUF: ModelInfo.ModelStorageFormat
    MODEL_STORAGE_FORMAT_PROTO_TEXT: ModelInfo.ModelStorageFormat
    MODEL_STORAGE_FORMAT_UNKNOWN: ModelInfo.ModelStorageFormat
    NAME_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSOR_SEMANTICS_FIELD_NUMBER: _ClassVar[int]
    bbox_info: BoundingBoxInfo
    class_labels: _containers.RepeatedScalarFieldContainer[str]
    file_assets: _containers.MessageMap[str, _uuid_pb2.UUID]
    format: ModelInfo.ModelStorageFormat
    image_preprocessing_steps: _containers.RepeatedCompositeFieldContainer[ImagePreprocessingStep]
    input_tensor_semantics: _containers.RepeatedCompositeFieldContainer[TensorSemantics]
    kind: ModelInfo.ModelKind
    name: str
    output_tensor_semantics: _containers.RepeatedCompositeFieldContainer[TensorSemantics]
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[_Union[ModelInfo.ModelKind, str]] = ..., format: _Optional[_Union[ModelInfo.ModelStorageFormat, str]] = ..., file_assets: _Optional[_Mapping[str, _uuid_pb2.UUID]] = ..., input_tensor_semantics: _Optional[_Iterable[_Union[TensorSemantics, _Mapping]]] = ..., output_tensor_semantics: _Optional[_Iterable[_Union[TensorSemantics, _Mapping]]] = ..., class_labels: _Optional[_Iterable[str]] = ..., bbox_info: _Optional[_Union[BoundingBoxInfo, _Mapping]] = ..., image_preprocessing_steps: _Optional[_Iterable[_Union[ImagePreprocessingStep, _Mapping]]] = ...) -> None: ...

class ModelSpec(_message.Message):
    __slots__ = ["name", "named_asset", "onnx_blob_key", "onnx_file", "openvino_spec", "runtime", "tensorrt_spec"]
    NAMED_ASSET_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ONNX_BLOB_KEY_FIELD_NUMBER: _ClassVar[int]
    ONNX_FILE_FIELD_NUMBER: _ClassVar[int]
    OPENVINO_SPEC_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    TENSORRT_SPEC_FIELD_NUMBER: _ClassVar[int]
    name: str
    named_asset: _containers.RepeatedCompositeFieldContainer[NamedAsset]
    onnx_blob_key: str
    onnx_file: FileResource
    openvino_spec: OpenVINOModelSpec
    runtime: str
    tensorrt_spec: TensorRTModelSpec
    def __init__(self, name: _Optional[str] = ..., onnx_blob_key: _Optional[str] = ..., onnx_file: _Optional[_Union[FileResource, _Mapping]] = ..., named_asset: _Optional[_Iterable[_Union[NamedAsset, _Mapping]]] = ..., runtime: _Optional[str] = ..., tensorrt_spec: _Optional[_Union[TensorRTModelSpec, _Mapping]] = ..., openvino_spec: _Optional[_Union[OpenVINOModelSpec, _Mapping]] = ...) -> None: ...

class NamedAsset(_message.Message):
    __slots__ = ["file", "name"]
    FILE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    file: FileResource
    name: str
    def __init__(self, name: _Optional[str] = ..., file: _Optional[_Union[FileResource, _Mapping]] = ...) -> None: ...

class Node(_message.Message):
    __slots__ = ["attributes", "inputs", "kind", "name", "outputs"]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    INPUTS_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    attributes: _containers.RepeatedCompositeFieldContainer[NodeAttributes]
    inputs: _containers.RepeatedCompositeFieldContainer[NodeInput]
    kind: LogicalPipelineNodeKind
    name: str
    outputs: _containers.RepeatedCompositeFieldContainer[NodeOutput]
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[_Union[LogicalPipelineNodeKind, str]] = ..., attributes: _Optional[_Iterable[_Union[NodeAttributes, _Mapping]]] = ..., inputs: _Optional[_Iterable[_Union[NodeInput, _Mapping]]] = ..., outputs: _Optional[_Iterable[_Union[NodeOutput, _Mapping]]] = ...) -> None: ...

class NodeAttributes(_message.Message):
    __slots__ = ["name", "value"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    name: str
    value: Value
    def __init__(self, name: _Optional[str] = ..., value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...

class NodeInput(_message.Message):
    __slots__ = ["lambda_input_value", "name", "node_output_value", "param_value"]
    class LambdaInputRef(_message.Message):
        __slots__ = ["name"]
        NAME_FIELD_NUMBER: _ClassVar[int]
        name: str
        def __init__(self, name: _Optional[str] = ...) -> None: ...
    class NodeOutputRef(_message.Message):
        __slots__ = ["name", "node_name"]
        NAME_FIELD_NUMBER: _ClassVar[int]
        NODE_NAME_FIELD_NUMBER: _ClassVar[int]
        name: str
        node_name: str
        def __init__(self, node_name: _Optional[str] = ..., name: _Optional[str] = ...) -> None: ...
    LAMBDA_INPUT_VALUE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    NODE_OUTPUT_VALUE_FIELD_NUMBER: _ClassVar[int]
    PARAM_VALUE_FIELD_NUMBER: _ClassVar[int]
    lambda_input_value: NodeInput.LambdaInputRef
    name: str
    node_output_value: NodeInput.NodeOutputRef
    param_value: ParamRef
    def __init__(self, name: _Optional[str] = ..., param_value: _Optional[_Union[ParamRef, _Mapping]] = ..., node_output_value: _Optional[_Union[NodeInput.NodeOutputRef, _Mapping]] = ..., lambda_input_value: _Optional[_Union[NodeInput.LambdaInputRef, _Mapping]] = ...) -> None: ...

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

class Param(_message.Message):
    __slots__ = ["default_value", "name"]
    DEFAULT_VALUE_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    default_value: Value
    name: str
    def __init__(self, name: _Optional[str] = ..., default_value: _Optional[_Union[Value, _Mapping]] = ...) -> None: ...

class ParamRef(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class PhysicalPipeline(_message.Message):
    __slots__ = ["created_at", "device_id", "id", "pipeline_deployment_id", "spec", "status", "updated_at"]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    PIPELINE_DEPLOYMENT_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    created_at: _timestamp_pb2.Timestamp
    device_id: _uuid_pb2.UUID
    id: _uuid_pb2.UUID
    pipeline_deployment_id: _uuid_pb2.UUID
    spec: PhysicalPipelineSpec
    status: PhysicalPipelineStatus
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., pipeline_deployment_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., spec: _Optional[_Union[PhysicalPipelineSpec, _Mapping]] = ..., status: _Optional[_Union[PhysicalPipelineStatus, _Mapping]] = ...) -> None: ...

class PhysicalPipelineSpec(_message.Message):
    __slots__ = ["graph", "runtime", "state", "version"]
    GRAPH_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    graph: ExecutionSpec
    runtime: str
    state: PipelineState
    version: int
    def __init__(self, state: _Optional[_Union[PipelineState, str]] = ..., version: _Optional[int] = ..., graph: _Optional[_Union[ExecutionSpec, _Mapping]] = ..., runtime: _Optional[str] = ...) -> None: ...

class PhysicalPipelineStatus(_message.Message):
    __slots__ = ["reason", "runtime", "state", "version"]
    REASON_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    reason: str
    runtime: str
    state: PipelineState
    version: int
    def __init__(self, state: _Optional[_Union[PipelineState, str]] = ..., version: _Optional[int] = ..., runtime: _Optional[str] = ..., reason: _Optional[str] = ...) -> None: ...

class PipelineDeployment(_message.Message):
    __slots__ = ["created_at", "deleted_at", "fleet_id", "id", "logical_pipeline_id", "spec", "status", "updated_at", "version"]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    DELETED_AT_FIELD_NUMBER: _ClassVar[int]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LOGICAL_PIPELINE_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    created_at: _timestamp_pb2.Timestamp
    deleted_at: _timestamp_pb2.Timestamp
    fleet_id: _uuid_pb2.UUID
    id: _uuid_pb2.UUID
    logical_pipeline_id: _uuid_pb2.UUID
    spec: PipelineDeploymentSpec
    status: PipelineDeploymentStatus
    updated_at: _timestamp_pb2.Timestamp
    version: int
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., logical_pipeline_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., fleet_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., version: _Optional[int] = ..., spec: _Optional[_Union[PipelineDeploymentSpec, _Mapping]] = ..., status: _Optional[_Union[PipelineDeploymentStatus, _Mapping]] = ..., deleted_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class PipelineDeploymentSpec(_message.Message):
    __slots__ = ["state"]
    STATE_FIELD_NUMBER: _ClassVar[int]
    state: PipelineState
    def __init__(self, state: _Optional[_Union[PipelineState, str]] = ...) -> None: ...

class PipelineDeploymentStatus(_message.Message):
    __slots__ = ["reason", "state"]
    REASON_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    reason: str
    state: PipelineState
    def __init__(self, state: _Optional[_Union[PipelineState, str]] = ..., reason: _Optional[str] = ...) -> None: ...

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

class TensorSemantics(_message.Message):
    __slots__ = ["dimensions"]
    DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    dimensions: _containers.RepeatedCompositeFieldContainer[DimensionSemantics]
    def __init__(self, dimensions: _Optional[_Iterable[_Union[DimensionSemantics, _Mapping]]] = ...) -> None: ...

class Value(_message.Message):
    __slots__ = ["bool_data", "double_data", "int64_data", "lambda_data", "model_data", "param_data", "string_data"]
    class Lambda(_message.Message):
        __slots__ = ["inputs", "nodes", "outputs"]
        INPUTS_FIELD_NUMBER: _ClassVar[int]
        NODES_FIELD_NUMBER: _ClassVar[int]
        OUTPUTS_FIELD_NUMBER: _ClassVar[int]
        inputs: _containers.RepeatedScalarFieldContainer[str]
        nodes: _containers.RepeatedCompositeFieldContainer[Node]
        outputs: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, inputs: _Optional[_Iterable[str]] = ..., outputs: _Optional[_Iterable[str]] = ..., nodes: _Optional[_Iterable[_Union[Node, _Mapping]]] = ...) -> None: ...
    class ModelRef(_message.Message):
        __slots__ = ["id", "name"]
        ID_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        id: _uuid_pb2.UUID
        name: str
        def __init__(self, name: _Optional[str] = ..., id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...
    BOOL_DATA_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_DATA_FIELD_NUMBER: _ClassVar[int]
    INT64_DATA_FIELD_NUMBER: _ClassVar[int]
    LAMBDA_DATA_FIELD_NUMBER: _ClassVar[int]
    MODEL_DATA_FIELD_NUMBER: _ClassVar[int]
    PARAM_DATA_FIELD_NUMBER: _ClassVar[int]
    STRING_DATA_FIELD_NUMBER: _ClassVar[int]
    bool_data: bool
    double_data: float
    int64_data: int
    lambda_data: Value.Lambda
    model_data: Value.ModelRef
    param_data: ParamRef
    string_data: str
    def __init__(self, string_data: _Optional[str] = ..., int64_data: _Optional[int] = ..., double_data: _Optional[float] = ..., bool_data: bool = ..., lambda_data: _Optional[_Union[Value.Lambda, _Mapping]] = ..., model_data: _Optional[_Union[Value.ModelRef, _Mapping]] = ..., param_data: _Optional[_Union[ParamRef, _Mapping]] = ...) -> None: ...

class LogicalPipelineNodeKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class PipelineState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
