from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class NonMaxSuppressionCalculatorOptions(_message.Message):
    __slots__ = ["algorithm", "max_num_detections", "min_score_threshold", "min_suppression_threshold", "num_detection_streams", "overlap_type", "return_empty_detections"]
    class NmsAlgorithm(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class OverlapType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    INTERSECTION_OVER_UNION: NonMaxSuppressionCalculatorOptions.OverlapType
    JACCARD: NonMaxSuppressionCalculatorOptions.OverlapType
    MAX_NUM_DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    MIN_SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MIN_SUPPRESSION_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MODIFIED_JACCARD: NonMaxSuppressionCalculatorOptions.OverlapType
    NMS_ALGO_DEFAULT: NonMaxSuppressionCalculatorOptions.NmsAlgorithm
    NMS_ALGO_WEIGHTED: NonMaxSuppressionCalculatorOptions.NmsAlgorithm
    NUM_DETECTION_STREAMS_FIELD_NUMBER: _ClassVar[int]
    OVERLAP_TYPE_FIELD_NUMBER: _ClassVar[int]
    RETURN_EMPTY_DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    UNSPECIFIED_OVERLAP_TYPE: NonMaxSuppressionCalculatorOptions.OverlapType
    algorithm: NonMaxSuppressionCalculatorOptions.NmsAlgorithm
    ext: _descriptor.FieldDescriptor
    max_num_detections: int
    min_score_threshold: float
    min_suppression_threshold: float
    num_detection_streams: int
    overlap_type: NonMaxSuppressionCalculatorOptions.OverlapType
    return_empty_detections: bool
    def __init__(self, num_detection_streams: _Optional[int] = ..., max_num_detections: _Optional[int] = ..., min_score_threshold: _Optional[float] = ..., min_suppression_threshold: _Optional[float] = ..., overlap_type: _Optional[_Union[NonMaxSuppressionCalculatorOptions.OverlapType, str]] = ..., return_empty_detections: bool = ..., algorithm: _Optional[_Union[NonMaxSuppressionCalculatorOptions.NmsAlgorithm, str]] = ...) -> None: ...
