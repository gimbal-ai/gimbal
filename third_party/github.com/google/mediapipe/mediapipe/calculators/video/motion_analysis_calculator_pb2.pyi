from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util.tracking import motion_analysis_pb2 as _motion_analysis_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class HomographyData(_message.Message):
    __slots__ = ["frame_height", "frame_width", "histogram_count_data", "motion_homography_data"]
    FRAME_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FRAME_WIDTH_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAM_COUNT_DATA_FIELD_NUMBER: _ClassVar[int]
    MOTION_HOMOGRAPHY_DATA_FIELD_NUMBER: _ClassVar[int]
    frame_height: int
    frame_width: int
    histogram_count_data: _containers.RepeatedScalarFieldContainer[int]
    motion_homography_data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, motion_homography_data: _Optional[_Iterable[float]] = ..., histogram_count_data: _Optional[_Iterable[int]] = ..., frame_width: _Optional[int] = ..., frame_height: _Optional[int] = ...) -> None: ...

class MotionAnalysisCalculatorOptions(_message.Message):
    __slots__ = ["analysis_options", "bypass_mode", "hybrid_selection_camera", "meta_analysis", "meta_models_per_frame", "meta_outlier_domain_ratio", "selection_analysis"]
    class MetaAnalysis(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class SelectionAnalysis(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ANALYSIS_FROM_FEATURES: MotionAnalysisCalculatorOptions.SelectionAnalysis
    ANALYSIS_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    ANALYSIS_RECOMPUTE: MotionAnalysisCalculatorOptions.SelectionAnalysis
    ANALYSIS_WITH_SEED: MotionAnalysisCalculatorOptions.SelectionAnalysis
    BYPASS_MODE_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    HYBRID_SELECTION_CAMERA_FIELD_NUMBER: _ClassVar[int]
    META_ANALYSIS_FIELD_NUMBER: _ClassVar[int]
    META_ANALYSIS_HYBRID: MotionAnalysisCalculatorOptions.MetaAnalysis
    META_ANALYSIS_USE_META: MotionAnalysisCalculatorOptions.MetaAnalysis
    META_MODELS_PER_FRAME_FIELD_NUMBER: _ClassVar[int]
    META_OUTLIER_DOMAIN_RATIO_FIELD_NUMBER: _ClassVar[int]
    NO_ANALYSIS_USE_SELECTION: MotionAnalysisCalculatorOptions.SelectionAnalysis
    SELECTION_ANALYSIS_FIELD_NUMBER: _ClassVar[int]
    analysis_options: _motion_analysis_pb2.MotionAnalysisOptions
    bypass_mode: bool
    ext: _descriptor.FieldDescriptor
    hybrid_selection_camera: bool
    meta_analysis: MotionAnalysisCalculatorOptions.MetaAnalysis
    meta_models_per_frame: int
    meta_outlier_domain_ratio: float
    selection_analysis: MotionAnalysisCalculatorOptions.SelectionAnalysis
    def __init__(self, analysis_options: _Optional[_Union[_motion_analysis_pb2.MotionAnalysisOptions, _Mapping]] = ..., selection_analysis: _Optional[_Union[MotionAnalysisCalculatorOptions.SelectionAnalysis, str]] = ..., hybrid_selection_camera: bool = ..., meta_analysis: _Optional[_Union[MotionAnalysisCalculatorOptions.MetaAnalysis, str]] = ..., meta_models_per_frame: _Optional[int] = ..., meta_outlier_domain_ratio: _Optional[float] = ..., bypass_mode: bool = ...) -> None: ...
