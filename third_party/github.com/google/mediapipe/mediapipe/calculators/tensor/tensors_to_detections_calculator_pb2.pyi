from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorsToDetectionsCalculatorOptions(_message.Message):
    __slots__ = ["allow_classes", "apply_exponential_on_box_size", "box_boundaries_indices", "box_coord_offset", "box_format", "flip_vertically", "h_scale", "ignore_classes", "keypoint_coord_offset", "max_results", "min_score_thresh", "num_boxes", "num_classes", "num_coords", "num_keypoints", "num_values_per_keypoint", "reverse_output_order", "score_clipping_thresh", "sigmoid_score", "tensor_mapping", "w_scale", "x_scale", "y_scale"]
    class BoxFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class BoxBoundariesIndices(_message.Message):
        __slots__ = ["xmax", "xmin", "ymax", "ymin"]
        XMAX_FIELD_NUMBER: _ClassVar[int]
        XMIN_FIELD_NUMBER: _ClassVar[int]
        YMAX_FIELD_NUMBER: _ClassVar[int]
        YMIN_FIELD_NUMBER: _ClassVar[int]
        xmax: int
        xmin: int
        ymax: int
        ymin: int
        def __init__(self, ymin: _Optional[int] = ..., xmin: _Optional[int] = ..., ymax: _Optional[int] = ..., xmax: _Optional[int] = ...) -> None: ...
    class TensorMapping(_message.Message):
        __slots__ = ["anchors_tensor_index", "classes_tensor_index", "detections_tensor_index", "num_detections_tensor_index", "scores_tensor_index"]
        ANCHORS_TENSOR_INDEX_FIELD_NUMBER: _ClassVar[int]
        CLASSES_TENSOR_INDEX_FIELD_NUMBER: _ClassVar[int]
        DETECTIONS_TENSOR_INDEX_FIELD_NUMBER: _ClassVar[int]
        NUM_DETECTIONS_TENSOR_INDEX_FIELD_NUMBER: _ClassVar[int]
        SCORES_TENSOR_INDEX_FIELD_NUMBER: _ClassVar[int]
        anchors_tensor_index: int
        classes_tensor_index: int
        detections_tensor_index: int
        num_detections_tensor_index: int
        scores_tensor_index: int
        def __init__(self, detections_tensor_index: _Optional[int] = ..., classes_tensor_index: _Optional[int] = ..., scores_tensor_index: _Optional[int] = ..., num_detections_tensor_index: _Optional[int] = ..., anchors_tensor_index: _Optional[int] = ...) -> None: ...
    ALLOW_CLASSES_FIELD_NUMBER: _ClassVar[int]
    APPLY_EXPONENTIAL_ON_BOX_SIZE_FIELD_NUMBER: _ClassVar[int]
    BOX_BOUNDARIES_INDICES_FIELD_NUMBER: _ClassVar[int]
    BOX_COORD_OFFSET_FIELD_NUMBER: _ClassVar[int]
    BOX_FORMAT_FIELD_NUMBER: _ClassVar[int]
    BOX_FORMAT_UNSPECIFIED: TensorsToDetectionsCalculatorOptions.BoxFormat
    BOX_FORMAT_XYWH: TensorsToDetectionsCalculatorOptions.BoxFormat
    BOX_FORMAT_XYXY: TensorsToDetectionsCalculatorOptions.BoxFormat
    BOX_FORMAT_YXHW: TensorsToDetectionsCalculatorOptions.BoxFormat
    EXT_FIELD_NUMBER: _ClassVar[int]
    FLIP_VERTICALLY_FIELD_NUMBER: _ClassVar[int]
    H_SCALE_FIELD_NUMBER: _ClassVar[int]
    IGNORE_CLASSES_FIELD_NUMBER: _ClassVar[int]
    KEYPOINT_COORD_OFFSET_FIELD_NUMBER: _ClassVar[int]
    MAX_RESULTS_FIELD_NUMBER: _ClassVar[int]
    MIN_SCORE_THRESH_FIELD_NUMBER: _ClassVar[int]
    NUM_BOXES_FIELD_NUMBER: _ClassVar[int]
    NUM_CLASSES_FIELD_NUMBER: _ClassVar[int]
    NUM_COORDS_FIELD_NUMBER: _ClassVar[int]
    NUM_KEYPOINTS_FIELD_NUMBER: _ClassVar[int]
    NUM_VALUES_PER_KEYPOINT_FIELD_NUMBER: _ClassVar[int]
    REVERSE_OUTPUT_ORDER_FIELD_NUMBER: _ClassVar[int]
    SCORE_CLIPPING_THRESH_FIELD_NUMBER: _ClassVar[int]
    SIGMOID_SCORE_FIELD_NUMBER: _ClassVar[int]
    TENSOR_MAPPING_FIELD_NUMBER: _ClassVar[int]
    W_SCALE_FIELD_NUMBER: _ClassVar[int]
    X_SCALE_FIELD_NUMBER: _ClassVar[int]
    Y_SCALE_FIELD_NUMBER: _ClassVar[int]
    allow_classes: _containers.RepeatedScalarFieldContainer[int]
    apply_exponential_on_box_size: bool
    box_boundaries_indices: TensorsToDetectionsCalculatorOptions.BoxBoundariesIndices
    box_coord_offset: int
    box_format: TensorsToDetectionsCalculatorOptions.BoxFormat
    ext: _descriptor.FieldDescriptor
    flip_vertically: bool
    h_scale: float
    ignore_classes: _containers.RepeatedScalarFieldContainer[int]
    keypoint_coord_offset: int
    max_results: int
    min_score_thresh: float
    num_boxes: int
    num_classes: int
    num_coords: int
    num_keypoints: int
    num_values_per_keypoint: int
    reverse_output_order: bool
    score_clipping_thresh: float
    sigmoid_score: bool
    tensor_mapping: TensorsToDetectionsCalculatorOptions.TensorMapping
    w_scale: float
    x_scale: float
    y_scale: float
    def __init__(self, num_classes: _Optional[int] = ..., num_boxes: _Optional[int] = ..., num_coords: _Optional[int] = ..., keypoint_coord_offset: _Optional[int] = ..., num_keypoints: _Optional[int] = ..., num_values_per_keypoint: _Optional[int] = ..., box_coord_offset: _Optional[int] = ..., x_scale: _Optional[float] = ..., y_scale: _Optional[float] = ..., w_scale: _Optional[float] = ..., h_scale: _Optional[float] = ..., apply_exponential_on_box_size: bool = ..., reverse_output_order: bool = ..., ignore_classes: _Optional[_Iterable[int]] = ..., allow_classes: _Optional[_Iterable[int]] = ..., sigmoid_score: bool = ..., score_clipping_thresh: _Optional[float] = ..., flip_vertically: bool = ..., min_score_thresh: _Optional[float] = ..., max_results: _Optional[int] = ..., tensor_mapping: _Optional[_Union[TensorsToDetectionsCalculatorOptions.TensorMapping, _Mapping]] = ..., box_boundaries_indices: _Optional[_Union[TensorsToDetectionsCalculatorOptions.BoxBoundariesIndices, _Mapping]] = ..., box_format: _Optional[_Union[TensorsToDetectionsCalculatorOptions.BoxFormat, str]] = ...) -> None: ...
