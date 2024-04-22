from mediapipe.util.tracking import motion_estimation_pb2 as _motion_estimation_pb2
from mediapipe.util.tracking import motion_saliency_pb2 as _motion_saliency_pb2
from mediapipe.util.tracking import region_flow_computation_pb2 as _region_flow_computation_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MotionAnalysisOptions(_message.Message):
    __slots__ = ["analysis_policy", "compute_motion_saliency", "estimation_clip_size", "filter_saliency", "flow_options", "foreground_options", "motion_options", "post_irls_smoothing", "rejection_transform_threshold", "saliency_options", "select_saliency_inliers", "subtract_camera_motion_from_features", "track_index", "visualization_options"]
    class AnalysisPolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class ForegroundOptions(_message.Message):
        __slots__ = ["foreground_gamma", "foreground_threshold", "threshold_coverage_scaling"]
        FOREGROUND_GAMMA_FIELD_NUMBER: _ClassVar[int]
        FOREGROUND_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        THRESHOLD_COVERAGE_SCALING_FIELD_NUMBER: _ClassVar[int]
        foreground_gamma: float
        foreground_threshold: float
        threshold_coverage_scaling: bool
        def __init__(self, foreground_threshold: _Optional[float] = ..., foreground_gamma: _Optional[float] = ..., threshold_coverage_scaling: bool = ...) -> None: ...
    class VisualizationOptions(_message.Message):
        __slots__ = ["foreground_jet_coloring", "line_thickness", "max_long_feature_points", "min_long_feature_track", "visualize_blur_analysis_region", "visualize_region_flow_features", "visualize_salient_points", "visualize_stats"]
        FOREGROUND_JET_COLORING_FIELD_NUMBER: _ClassVar[int]
        LINE_THICKNESS_FIELD_NUMBER: _ClassVar[int]
        MAX_LONG_FEATURE_POINTS_FIELD_NUMBER: _ClassVar[int]
        MIN_LONG_FEATURE_TRACK_FIELD_NUMBER: _ClassVar[int]
        VISUALIZE_BLUR_ANALYSIS_REGION_FIELD_NUMBER: _ClassVar[int]
        VISUALIZE_REGION_FLOW_FEATURES_FIELD_NUMBER: _ClassVar[int]
        VISUALIZE_SALIENT_POINTS_FIELD_NUMBER: _ClassVar[int]
        VISUALIZE_STATS_FIELD_NUMBER: _ClassVar[int]
        foreground_jet_coloring: bool
        line_thickness: int
        max_long_feature_points: int
        min_long_feature_track: int
        visualize_blur_analysis_region: bool
        visualize_region_flow_features: bool
        visualize_salient_points: bool
        visualize_stats: bool
        def __init__(self, visualize_region_flow_features: bool = ..., visualize_salient_points: bool = ..., line_thickness: _Optional[int] = ..., foreground_jet_coloring: bool = ..., visualize_blur_analysis_region: bool = ..., visualize_stats: bool = ..., min_long_feature_track: _Optional[int] = ..., max_long_feature_points: _Optional[int] = ...) -> None: ...
    ANALYSIS_POLICY_CAMERA_MOBILE: MotionAnalysisOptions.AnalysisPolicy
    ANALYSIS_POLICY_FIELD_NUMBER: _ClassVar[int]
    ANALYSIS_POLICY_HYPERLAPSE: MotionAnalysisOptions.AnalysisPolicy
    ANALYSIS_POLICY_LEGACY: MotionAnalysisOptions.AnalysisPolicy
    ANALYSIS_POLICY_VIDEO: MotionAnalysisOptions.AnalysisPolicy
    ANALYSIS_POLICY_VIDEO_MOBILE: MotionAnalysisOptions.AnalysisPolicy
    COMPUTE_MOTION_SALIENCY_FIELD_NUMBER: _ClassVar[int]
    ESTIMATION_CLIP_SIZE_FIELD_NUMBER: _ClassVar[int]
    FILTER_SALIENCY_FIELD_NUMBER: _ClassVar[int]
    FLOW_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    FOREGROUND_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    MOTION_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    POST_IRLS_SMOOTHING_FIELD_NUMBER: _ClassVar[int]
    REJECTION_TRANSFORM_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SALIENCY_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    SELECT_SALIENCY_INLIERS_FIELD_NUMBER: _ClassVar[int]
    SUBTRACT_CAMERA_MOTION_FROM_FEATURES_FIELD_NUMBER: _ClassVar[int]
    TRACK_INDEX_FIELD_NUMBER: _ClassVar[int]
    VISUALIZATION_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    analysis_policy: MotionAnalysisOptions.AnalysisPolicy
    compute_motion_saliency: bool
    estimation_clip_size: int
    filter_saliency: bool
    flow_options: _region_flow_computation_pb2.RegionFlowComputationOptions
    foreground_options: MotionAnalysisOptions.ForegroundOptions
    motion_options: _motion_estimation_pb2.MotionEstimationOptions
    post_irls_smoothing: bool
    rejection_transform_threshold: float
    saliency_options: _motion_saliency_pb2.MotionSaliencyOptions
    select_saliency_inliers: bool
    subtract_camera_motion_from_features: bool
    track_index: int
    visualization_options: MotionAnalysisOptions.VisualizationOptions
    def __init__(self, analysis_policy: _Optional[_Union[MotionAnalysisOptions.AnalysisPolicy, str]] = ..., flow_options: _Optional[_Union[_region_flow_computation_pb2.RegionFlowComputationOptions, _Mapping]] = ..., motion_options: _Optional[_Union[_motion_estimation_pb2.MotionEstimationOptions, _Mapping]] = ..., saliency_options: _Optional[_Union[_motion_saliency_pb2.MotionSaliencyOptions, _Mapping]] = ..., estimation_clip_size: _Optional[int] = ..., subtract_camera_motion_from_features: bool = ..., track_index: _Optional[int] = ..., compute_motion_saliency: bool = ..., select_saliency_inliers: bool = ..., filter_saliency: bool = ..., post_irls_smoothing: bool = ..., rejection_transform_threshold: _Optional[float] = ..., visualization_options: _Optional[_Union[MotionAnalysisOptions.VisualizationOptions, _Mapping]] = ..., foreground_options: _Optional[_Union[MotionAnalysisOptions.ForegroundOptions, _Mapping]] = ...) -> None: ...
