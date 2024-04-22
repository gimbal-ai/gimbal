from mediapipe.util.tracking import tone_estimation_pb2 as _tone_estimation_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RegionFlowComputationOptions(_message.Message):
    __slots__ = ["absolute_inlier_error_threshold", "blur_score_options", "compute_blur_score", "compute_derivative_in_pyramid", "corner_response_scale", "descriptor_extractor_type", "distance_from_border", "downsample_factor", "downsample_mode", "downsample_schedule", "downsampling_size", "fast_estimation_block_size", "fast_estimation_min_block_size", "fast_estimation_overlap_grids", "fast_gain_correction", "frac_gain_feature_size", "frac_gain_step", "frac_inlier_error_threshold", "gain_bias_bounds", "gain_correct_mode", "gain_correction", "gain_correction_bright_reference", "gain_correction_inlier_improvement_frac", "gain_correction_multiple_hypotheses", "gain_correction_triggering_ratio", "histogram_equalization", "image_format", "irls_initialization", "long_feature_verification_threshold", "max_long_feature_acceleration", "max_magnitude_threshold_ratio", "median_magnitude_bounds", "min_feature_cover", "min_feature_cover_grid", "min_feature_inliers", "min_feature_requirement", "no_estimation_mode", "patch_descriptor_radius", "pre_blur_sigma", "ransac_rounds_per_region", "relative_inlier_error_threshold", "relative_min_feature_inliers", "round_downsample_factor", "top_inlier_sets", "tracking_options", "use_synthetic_zero_motion_tracks_all_frames", "use_synthetic_zero_motion_tracks_first_frame", "verification_distance", "verify_features", "verify_long_feature_acceleration", "verify_long_feature_trigger_ratio", "verify_long_features", "visual_consistency_options"]
    class DescriptorExtractorType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class DownsampleMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class GainCorrectMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class ImageFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class IrlsInitialization(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class BlurScoreOptions(_message.Message):
        __slots__ = ["absolute_cornerness_threshold", "box_filter_diam", "median_percentile", "relative_cornerness_threshold"]
        ABSOLUTE_CORNERNESS_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        BOX_FILTER_DIAM_FIELD_NUMBER: _ClassVar[int]
        MEDIAN_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
        RELATIVE_CORNERNESS_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        absolute_cornerness_threshold: float
        box_filter_diam: int
        median_percentile: float
        relative_cornerness_threshold: float
        def __init__(self, box_filter_diam: _Optional[int] = ..., relative_cornerness_threshold: _Optional[float] = ..., absolute_cornerness_threshold: _Optional[float] = ..., median_percentile: _Optional[float] = ...) -> None: ...
    class DownSampleSchedule(_message.Message):
        __slots__ = ["downsample_factor_1080p", "downsample_factor_360p", "downsample_factor_480p", "downsample_factor_720p"]
        DOWNSAMPLE_FACTOR_1080P_FIELD_NUMBER: _ClassVar[int]
        DOWNSAMPLE_FACTOR_360P_FIELD_NUMBER: _ClassVar[int]
        DOWNSAMPLE_FACTOR_480P_FIELD_NUMBER: _ClassVar[int]
        DOWNSAMPLE_FACTOR_720P_FIELD_NUMBER: _ClassVar[int]
        downsample_factor_1080p: float
        downsample_factor_360p: float
        downsample_factor_480p: float
        downsample_factor_720p: float
        def __init__(self, downsample_factor_360p: _Optional[float] = ..., downsample_factor_480p: _Optional[float] = ..., downsample_factor_720p: _Optional[float] = ..., downsample_factor_1080p: _Optional[float] = ...) -> None: ...
    class VisualConsistencyOptions(_message.Message):
        __slots__ = ["compute_consistency", "tiny_image_dimension"]
        COMPUTE_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
        TINY_IMAGE_DIMENSION_FIELD_NUMBER: _ClassVar[int]
        compute_consistency: bool
        tiny_image_dimension: int
        def __init__(self, compute_consistency: bool = ..., tiny_image_dimension: _Optional[int] = ...) -> None: ...
    ABSOLUTE_INLIER_ERROR_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    BLUR_SCORE_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_BLUR_SCORE_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_DERIVATIVE_IN_PYRAMID_FIELD_NUMBER: _ClassVar[int]
    CORNER_RESPONSE_SCALE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTOR_EXTRACTOR_ORB: RegionFlowComputationOptions.DescriptorExtractorType
    DESCRIPTOR_EXTRACTOR_TYPE_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FROM_BORDER_FIELD_NUMBER: _ClassVar[int]
    DOWNSAMPLE_FACTOR_FIELD_NUMBER: _ClassVar[int]
    DOWNSAMPLE_MODE_BY_FACTOR: RegionFlowComputationOptions.DownsampleMode
    DOWNSAMPLE_MODE_BY_SCHEDULE: RegionFlowComputationOptions.DownsampleMode
    DOWNSAMPLE_MODE_FIELD_NUMBER: _ClassVar[int]
    DOWNSAMPLE_MODE_NONE: RegionFlowComputationOptions.DownsampleMode
    DOWNSAMPLE_MODE_TO_INPUT_SIZE: RegionFlowComputationOptions.DownsampleMode
    DOWNSAMPLE_MODE_TO_MAX_SIZE: RegionFlowComputationOptions.DownsampleMode
    DOWNSAMPLE_MODE_TO_MIN_SIZE: RegionFlowComputationOptions.DownsampleMode
    DOWNSAMPLE_MODE_UNKNOWN: RegionFlowComputationOptions.DownsampleMode
    DOWNSAMPLE_SCHEDULE_FIELD_NUMBER: _ClassVar[int]
    DOWNSAMPLING_SIZE_FIELD_NUMBER: _ClassVar[int]
    Extensions: _python_message._ExtensionDict
    FAST_ESTIMATION_BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    FAST_ESTIMATION_MIN_BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    FAST_ESTIMATION_OVERLAP_GRIDS_FIELD_NUMBER: _ClassVar[int]
    FAST_GAIN_CORRECTION_FIELD_NUMBER: _ClassVar[int]
    FRAC_GAIN_FEATURE_SIZE_FIELD_NUMBER: _ClassVar[int]
    FRAC_GAIN_STEP_FIELD_NUMBER: _ClassVar[int]
    FRAC_INLIER_ERROR_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    GAIN_BIAS_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    GAIN_CORRECTION_BRIGHT_REFERENCE_FIELD_NUMBER: _ClassVar[int]
    GAIN_CORRECTION_FIELD_NUMBER: _ClassVar[int]
    GAIN_CORRECTION_INLIER_IMPROVEMENT_FRAC_FIELD_NUMBER: _ClassVar[int]
    GAIN_CORRECTION_MULTIPLE_HYPOTHESES_FIELD_NUMBER: _ClassVar[int]
    GAIN_CORRECTION_TRIGGERING_RATIO_FIELD_NUMBER: _ClassVar[int]
    GAIN_CORRECT_MODE_DEFAULT_USER: RegionFlowComputationOptions.GainCorrectMode
    GAIN_CORRECT_MODE_FIELD_NUMBER: _ClassVar[int]
    GAIN_CORRECT_MODE_HDR: RegionFlowComputationOptions.GainCorrectMode
    GAIN_CORRECT_MODE_PHOTO_BURST: RegionFlowComputationOptions.GainCorrectMode
    GAIN_CORRECT_MODE_VIDEO: RegionFlowComputationOptions.GainCorrectMode
    HISTOGRAM_EQUALIZATION_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FORMAT_BGR: RegionFlowComputationOptions.ImageFormat
    IMAGE_FORMAT_BGRA: RegionFlowComputationOptions.ImageFormat
    IMAGE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FORMAT_GRAYSCALE: RegionFlowComputationOptions.ImageFormat
    IMAGE_FORMAT_RGB: RegionFlowComputationOptions.ImageFormat
    IMAGE_FORMAT_RGBA: RegionFlowComputationOptions.ImageFormat
    IMAGE_FORMAT_UNKNOWN: RegionFlowComputationOptions.ImageFormat
    IRIS_INIT_CONSISTENCY: RegionFlowComputationOptions.IrlsInitialization
    IRIS_INIT_UNIFORM: RegionFlowComputationOptions.IrlsInitialization
    IRIS_INIT_UNKNOWN: RegionFlowComputationOptions.IrlsInitialization
    IRLS_INITIALIZATION_FIELD_NUMBER: _ClassVar[int]
    LONG_FEATURE_VERIFICATION_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_LONG_FEATURE_ACCELERATION_FIELD_NUMBER: _ClassVar[int]
    MAX_MAGNITUDE_THRESHOLD_RATIO_FIELD_NUMBER: _ClassVar[int]
    MEDIAN_MAGNITUDE_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    MIN_FEATURE_COVER_FIELD_NUMBER: _ClassVar[int]
    MIN_FEATURE_COVER_GRID_FIELD_NUMBER: _ClassVar[int]
    MIN_FEATURE_INLIERS_FIELD_NUMBER: _ClassVar[int]
    MIN_FEATURE_REQUIREMENT_FIELD_NUMBER: _ClassVar[int]
    NO_ESTIMATION_MODE_FIELD_NUMBER: _ClassVar[int]
    PATCH_DESCRIPTOR_RADIUS_FIELD_NUMBER: _ClassVar[int]
    PRE_BLUR_SIGMA_FIELD_NUMBER: _ClassVar[int]
    RANSAC_ROUNDS_PER_REGION_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_INLIER_ERROR_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_MIN_FEATURE_INLIERS_FIELD_NUMBER: _ClassVar[int]
    ROUND_DOWNSAMPLE_FACTOR_FIELD_NUMBER: _ClassVar[int]
    TOP_INLIER_SETS_FIELD_NUMBER: _ClassVar[int]
    TRACKING_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    USE_SYNTHETIC_ZERO_MOTION_TRACKS_ALL_FRAMES_FIELD_NUMBER: _ClassVar[int]
    USE_SYNTHETIC_ZERO_MOTION_TRACKS_FIRST_FRAME_FIELD_NUMBER: _ClassVar[int]
    VERIFICATION_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    VERIFY_FEATURES_FIELD_NUMBER: _ClassVar[int]
    VERIFY_LONG_FEATURES_FIELD_NUMBER: _ClassVar[int]
    VERIFY_LONG_FEATURE_ACCELERATION_FIELD_NUMBER: _ClassVar[int]
    VERIFY_LONG_FEATURE_TRIGGER_RATIO_FIELD_NUMBER: _ClassVar[int]
    VISUAL_CONSISTENCY_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    absolute_inlier_error_threshold: float
    blur_score_options: RegionFlowComputationOptions.BlurScoreOptions
    compute_blur_score: bool
    compute_derivative_in_pyramid: bool
    corner_response_scale: float
    descriptor_extractor_type: RegionFlowComputationOptions.DescriptorExtractorType
    distance_from_border: int
    downsample_factor: float
    downsample_mode: RegionFlowComputationOptions.DownsampleMode
    downsample_schedule: RegionFlowComputationOptions.DownSampleSchedule
    downsampling_size: int
    fast_estimation_block_size: float
    fast_estimation_min_block_size: int
    fast_estimation_overlap_grids: int
    fast_gain_correction: bool
    frac_gain_feature_size: float
    frac_gain_step: float
    frac_inlier_error_threshold: float
    gain_bias_bounds: _tone_estimation_pb2.ToneEstimationOptions.GainBiasBounds
    gain_correct_mode: RegionFlowComputationOptions.GainCorrectMode
    gain_correction: bool
    gain_correction_bright_reference: bool
    gain_correction_inlier_improvement_frac: float
    gain_correction_multiple_hypotheses: bool
    gain_correction_triggering_ratio: float
    histogram_equalization: bool
    image_format: RegionFlowComputationOptions.ImageFormat
    irls_initialization: RegionFlowComputationOptions.IrlsInitialization
    long_feature_verification_threshold: float
    max_long_feature_acceleration: float
    max_magnitude_threshold_ratio: float
    median_magnitude_bounds: float
    min_feature_cover: float
    min_feature_cover_grid: int
    min_feature_inliers: int
    min_feature_requirement: int
    no_estimation_mode: bool
    patch_descriptor_radius: int
    pre_blur_sigma: float
    ransac_rounds_per_region: int
    relative_inlier_error_threshold: float
    relative_min_feature_inliers: float
    round_downsample_factor: bool
    top_inlier_sets: int
    tracking_options: TrackingOptions
    use_synthetic_zero_motion_tracks_all_frames: bool
    use_synthetic_zero_motion_tracks_first_frame: bool
    verification_distance: float
    verify_features: bool
    verify_long_feature_acceleration: bool
    verify_long_feature_trigger_ratio: float
    verify_long_features: bool
    visual_consistency_options: RegionFlowComputationOptions.VisualConsistencyOptions
    def __init__(self, tracking_options: _Optional[_Union[TrackingOptions, _Mapping]] = ..., min_feature_inliers: _Optional[int] = ..., relative_min_feature_inliers: _Optional[float] = ..., pre_blur_sigma: _Optional[float] = ..., ransac_rounds_per_region: _Optional[int] = ..., absolute_inlier_error_threshold: _Optional[float] = ..., frac_inlier_error_threshold: _Optional[float] = ..., relative_inlier_error_threshold: _Optional[float] = ..., top_inlier_sets: _Optional[int] = ..., no_estimation_mode: bool = ..., fast_estimation_block_size: _Optional[float] = ..., fast_estimation_min_block_size: _Optional[int] = ..., fast_estimation_overlap_grids: _Optional[int] = ..., max_magnitude_threshold_ratio: _Optional[float] = ..., median_magnitude_bounds: _Optional[float] = ..., irls_initialization: _Optional[_Union[RegionFlowComputationOptions.IrlsInitialization, str]] = ..., downsample_mode: _Optional[_Union[RegionFlowComputationOptions.DownsampleMode, str]] = ..., downsampling_size: _Optional[int] = ..., downsample_factor: _Optional[float] = ..., round_downsample_factor: bool = ..., downsample_schedule: _Optional[_Union[RegionFlowComputationOptions.DownSampleSchedule, _Mapping]] = ..., min_feature_requirement: _Optional[int] = ..., min_feature_cover: _Optional[float] = ..., min_feature_cover_grid: _Optional[int] = ..., compute_blur_score: bool = ..., blur_score_options: _Optional[_Union[RegionFlowComputationOptions.BlurScoreOptions, _Mapping]] = ..., visual_consistency_options: _Optional[_Union[RegionFlowComputationOptions.VisualConsistencyOptions, _Mapping]] = ..., patch_descriptor_radius: _Optional[int] = ..., distance_from_border: _Optional[int] = ..., corner_response_scale: _Optional[float] = ..., verify_features: bool = ..., verification_distance: _Optional[float] = ..., verify_long_features: bool = ..., long_feature_verification_threshold: _Optional[float] = ..., max_long_feature_acceleration: _Optional[float] = ..., verify_long_feature_acceleration: bool = ..., verify_long_feature_trigger_ratio: _Optional[float] = ..., histogram_equalization: bool = ..., use_synthetic_zero_motion_tracks_all_frames: bool = ..., use_synthetic_zero_motion_tracks_first_frame: bool = ..., gain_correction: bool = ..., fast_gain_correction: bool = ..., gain_correction_multiple_hypotheses: bool = ..., gain_correction_inlier_improvement_frac: _Optional[float] = ..., gain_correction_bright_reference: bool = ..., gain_correction_triggering_ratio: _Optional[float] = ..., frac_gain_feature_size: _Optional[float] = ..., frac_gain_step: _Optional[float] = ..., gain_correct_mode: _Optional[_Union[RegionFlowComputationOptions.GainCorrectMode, str]] = ..., gain_bias_bounds: _Optional[_Union[_tone_estimation_pb2.ToneEstimationOptions.GainBiasBounds, _Mapping]] = ..., image_format: _Optional[_Union[RegionFlowComputationOptions.ImageFormat, str]] = ..., descriptor_extractor_type: _Optional[_Union[RegionFlowComputationOptions.DescriptorExtractorType, str]] = ..., compute_derivative_in_pyramid: bool = ...) -> None: ...

class TrackingOptions(_message.Message):
    __slots__ = ["adaptive_extraction_levels", "adaptive_extraction_levels_lowest_size", "adaptive_features_block_size", "adaptive_features_levels", "adaptive_good_features_to_track", "adaptive_tracking_distance", "corner_extraction_method", "distance_downscale_sqrt", "fast_settings", "fractional_tracking_distance", "harris_settings", "internal_tracking_direction", "klt_tracker_implementation", "long_tracks_max_frames", "max_features", "min_eig_val_settings", "min_feature_distance", "multi_frames_to_track", "output_flow_direction", "ratio_test_threshold", "refine_wide_baseline_matches", "reuse_features_max_frame_distance", "reuse_features_min_survived_frac", "synthetic_zero_motion_grid_step", "tracking_iterations", "tracking_policy", "tracking_window_size", "wide_baseline_matching"]
    class CornerExtractionMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class FlowDirection(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class KltTrackerImplementation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class TrackingPolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class FastExtractionSettings(_message.Message):
        __slots__ = ["threshold"]
        THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        threshold: int
        def __init__(self, threshold: _Optional[int] = ...) -> None: ...
    class HarrisExtractionSettings(_message.Message):
        __slots__ = ["feature_quality_level"]
        FEATURE_QUALITY_LEVEL_FIELD_NUMBER: _ClassVar[int]
        feature_quality_level: float
        def __init__(self, feature_quality_level: _Optional[float] = ...) -> None: ...
    class MinEigValExtractionSettings(_message.Message):
        __slots__ = ["adaptive_lowest_quality_level", "feature_quality_level"]
        ADAPTIVE_LOWEST_QUALITY_LEVEL_FIELD_NUMBER: _ClassVar[int]
        FEATURE_QUALITY_LEVEL_FIELD_NUMBER: _ClassVar[int]
        adaptive_lowest_quality_level: float
        feature_quality_level: float
        def __init__(self, feature_quality_level: _Optional[float] = ..., adaptive_lowest_quality_level: _Optional[float] = ...) -> None: ...
    ADAPTIVE_EXTRACTION_LEVELS_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_EXTRACTION_LEVELS_LOWEST_SIZE_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_FEATURES_BLOCK_SIZE_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_FEATURES_LEVELS_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_GOOD_FEATURES_TO_TRACK_FIELD_NUMBER: _ClassVar[int]
    ADAPTIVE_TRACKING_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    CORNER_EXTRACTION_METHOD_FAST: TrackingOptions.CornerExtractionMethod
    CORNER_EXTRACTION_METHOD_FIELD_NUMBER: _ClassVar[int]
    CORNER_EXTRACTION_METHOD_HARRIS: TrackingOptions.CornerExtractionMethod
    CORNER_EXTRACTION_METHOD_MIN_EIG_VAL: TrackingOptions.CornerExtractionMethod
    CORNER_EXTRACTION_METHOD_UNKNOWN: TrackingOptions.CornerExtractionMethod
    DISTANCE_DOWNSCALE_SQRT_FIELD_NUMBER: _ClassVar[int]
    Extensions: _python_message._ExtensionDict
    FAST_SETTINGS_FIELD_NUMBER: _ClassVar[int]
    FLOW_DIRECTION_BACKWARD: TrackingOptions.FlowDirection
    FLOW_DIRECTION_CONSECUTIVELY: TrackingOptions.FlowDirection
    FLOW_DIRECTION_FORWARD: TrackingOptions.FlowDirection
    FLOW_DIRECTION_UNKNOWN: TrackingOptions.FlowDirection
    FRACTIONAL_TRACKING_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    HARRIS_SETTINGS_FIELD_NUMBER: _ClassVar[int]
    INTERNAL_TRACKING_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    KLT_OPENCV: TrackingOptions.KltTrackerImplementation
    KLT_TRACKER_IMPLEMENTATION_FIELD_NUMBER: _ClassVar[int]
    KLT_UNSPECIFIED: TrackingOptions.KltTrackerImplementation
    LONG_TRACKS_MAX_FRAMES_FIELD_NUMBER: _ClassVar[int]
    MAX_FEATURES_FIELD_NUMBER: _ClassVar[int]
    MIN_EIG_VAL_SETTINGS_FIELD_NUMBER: _ClassVar[int]
    MIN_FEATURE_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    MULTI_FRAMES_TO_TRACK_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FLOW_DIRECTION_FIELD_NUMBER: _ClassVar[int]
    RATIO_TEST_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    REFINE_WIDE_BASELINE_MATCHES_FIELD_NUMBER: _ClassVar[int]
    REUSE_FEATURES_MAX_FRAME_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    REUSE_FEATURES_MIN_SURVIVED_FRAC_FIELD_NUMBER: _ClassVar[int]
    SYNTHETIC_ZERO_MOTION_GRID_STEP_FIELD_NUMBER: _ClassVar[int]
    TRACKING_ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    TRACKING_POLICY_FIELD_NUMBER: _ClassVar[int]
    TRACKING_POLICY_LONG_TRACKS: TrackingOptions.TrackingPolicy
    TRACKING_POLICY_MULTI_FRAME: TrackingOptions.TrackingPolicy
    TRACKING_POLICY_SINGLE_FRAME: TrackingOptions.TrackingPolicy
    TRACKING_POLICY_UNKNOWN: TrackingOptions.TrackingPolicy
    TRACKING_WINDOW_SIZE_FIELD_NUMBER: _ClassVar[int]
    WIDE_BASELINE_MATCHING_FIELD_NUMBER: _ClassVar[int]
    adaptive_extraction_levels: int
    adaptive_extraction_levels_lowest_size: int
    adaptive_features_block_size: float
    adaptive_features_levels: int
    adaptive_good_features_to_track: bool
    adaptive_tracking_distance: bool
    corner_extraction_method: TrackingOptions.CornerExtractionMethod
    distance_downscale_sqrt: bool
    fast_settings: TrackingOptions.FastExtractionSettings
    fractional_tracking_distance: float
    harris_settings: TrackingOptions.HarrisExtractionSettings
    internal_tracking_direction: TrackingOptions.FlowDirection
    klt_tracker_implementation: TrackingOptions.KltTrackerImplementation
    long_tracks_max_frames: int
    max_features: int
    min_eig_val_settings: TrackingOptions.MinEigValExtractionSettings
    min_feature_distance: float
    multi_frames_to_track: int
    output_flow_direction: TrackingOptions.FlowDirection
    ratio_test_threshold: float
    refine_wide_baseline_matches: bool
    reuse_features_max_frame_distance: int
    reuse_features_min_survived_frac: float
    synthetic_zero_motion_grid_step: float
    tracking_iterations: int
    tracking_policy: TrackingOptions.TrackingPolicy
    tracking_window_size: int
    wide_baseline_matching: bool
    def __init__(self, internal_tracking_direction: _Optional[_Union[TrackingOptions.FlowDirection, str]] = ..., output_flow_direction: _Optional[_Union[TrackingOptions.FlowDirection, str]] = ..., tracking_policy: _Optional[_Union[TrackingOptions.TrackingPolicy, str]] = ..., multi_frames_to_track: _Optional[int] = ..., long_tracks_max_frames: _Optional[int] = ..., max_features: _Optional[int] = ..., corner_extraction_method: _Optional[_Union[TrackingOptions.CornerExtractionMethod, str]] = ..., min_eig_val_settings: _Optional[_Union[TrackingOptions.MinEigValExtractionSettings, _Mapping]] = ..., harris_settings: _Optional[_Union[TrackingOptions.HarrisExtractionSettings, _Mapping]] = ..., fast_settings: _Optional[_Union[TrackingOptions.FastExtractionSettings, _Mapping]] = ..., tracking_window_size: _Optional[int] = ..., tracking_iterations: _Optional[int] = ..., fractional_tracking_distance: _Optional[float] = ..., adaptive_tracking_distance: bool = ..., min_feature_distance: _Optional[float] = ..., distance_downscale_sqrt: bool = ..., adaptive_good_features_to_track: bool = ..., adaptive_features_block_size: _Optional[float] = ..., adaptive_features_levels: _Optional[int] = ..., adaptive_extraction_levels: _Optional[int] = ..., adaptive_extraction_levels_lowest_size: _Optional[int] = ..., synthetic_zero_motion_grid_step: _Optional[float] = ..., wide_baseline_matching: bool = ..., ratio_test_threshold: _Optional[float] = ..., refine_wide_baseline_matches: bool = ..., reuse_features_max_frame_distance: _Optional[int] = ..., reuse_features_min_survived_frac: _Optional[float] = ..., klt_tracker_implementation: _Optional[_Union[TrackingOptions.KltTrackerImplementation, str]] = ...) -> None: ...
