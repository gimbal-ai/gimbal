from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MotionEstimationOptions(_message.Message):
    __slots__ = ["affine_estimation", "coverage_grid_size", "deactivate_stable_motion_estimation", "domain_limited_irls_scaling", "estimate_similarity", "estimate_translation_irls", "estimation_policy", "feature_density_normalization", "feature_grid_size", "feature_mask_size", "feature_sigma", "filter_5_taps", "filter_initialized_irls_weights", "frame_confidence_weighting", "homography_estimation", "homography_exact_denominator_scaling", "homography_irls_weight_initialization", "homography_perspective_regularizer", "irls_initialization", "irls_mask_options", "irls_mixture_fraction_scale", "irls_motion_magnitude_fraction", "irls_prior_scale", "irls_rounds", "irls_use_l0_norm", "irls_weight_filter", "irls_weights_preinitialized", "joint_track_estimation", "label_empty_frames_as_valid", "lin_sim_inlier_threshold", "linear_similarity_estimation", "long_feature_bias_options", "long_feature_initialization", "mix_homography_estimation", "mixture_model_mode", "mixture_regularizer", "mixture_regularizer_base", "mixture_regularizer_levels", "mixture_row_sigma", "mixture_rs_analysis_level", "num_mixtures", "output_refined_irls_weights", "overlay_analysis_chunk_size", "overlay_detection", "overlay_detection_options", "project_valid_motions_down", "reset_confidence_threshold", "shot_boundary_options", "spatial_sigma", "stable_homography_bounds", "stable_mixture_homography_bounds", "stable_similarity_bounds", "stable_translation_bounds", "strict_coverage_scale", "temporal_irls_diameter", "temporal_sigma", "use_exact_homography_estimation", "use_highest_accuracy_for_normal_equations", "use_only_lin_sim_inliers_for_homography"]
    class AffineEstimation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class EstimationPolicy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class HomographyEstimation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class HomographyIrlsWeightInitialization(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class IRLSWeightFilter(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class LinearSimilarityEstimation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class MixtureHomographyEstimation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class MixtureModelMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class HomographyBounds(_message.Message):
        __slots__ = ["frac_inlier_threshold", "frac_registration_threshold", "limit_perspective", "limit_rotation", "lower_scale", "min_inlier_coverage", "registration_threshold", "upper_scale"]
        FRAC_INLIER_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        FRAC_REGISTRATION_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        LIMIT_PERSPECTIVE_FIELD_NUMBER: _ClassVar[int]
        LIMIT_ROTATION_FIELD_NUMBER: _ClassVar[int]
        LOWER_SCALE_FIELD_NUMBER: _ClassVar[int]
        MIN_INLIER_COVERAGE_FIELD_NUMBER: _ClassVar[int]
        REGISTRATION_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        UPPER_SCALE_FIELD_NUMBER: _ClassVar[int]
        frac_inlier_threshold: float
        frac_registration_threshold: float
        limit_perspective: float
        limit_rotation: float
        lower_scale: float
        min_inlier_coverage: float
        registration_threshold: float
        upper_scale: float
        def __init__(self, lower_scale: _Optional[float] = ..., upper_scale: _Optional[float] = ..., limit_rotation: _Optional[float] = ..., limit_perspective: _Optional[float] = ..., registration_threshold: _Optional[float] = ..., frac_registration_threshold: _Optional[float] = ..., min_inlier_coverage: _Optional[float] = ..., frac_inlier_threshold: _Optional[float] = ...) -> None: ...
    class IrlsMaskOptions(_message.Message):
        __slots__ = ["base_score", "decay", "inlier_score", "min_translation_norm", "translation_blend_alpha", "translation_prior_increase"]
        BASE_SCORE_FIELD_NUMBER: _ClassVar[int]
        DECAY_FIELD_NUMBER: _ClassVar[int]
        Extensions: _python_message._ExtensionDict
        INLIER_SCORE_FIELD_NUMBER: _ClassVar[int]
        MIN_TRANSLATION_NORM_FIELD_NUMBER: _ClassVar[int]
        TRANSLATION_BLEND_ALPHA_FIELD_NUMBER: _ClassVar[int]
        TRANSLATION_PRIOR_INCREASE_FIELD_NUMBER: _ClassVar[int]
        base_score: float
        decay: float
        inlier_score: float
        min_translation_norm: float
        translation_blend_alpha: float
        translation_prior_increase: float
        def __init__(self, decay: _Optional[float] = ..., inlier_score: _Optional[float] = ..., base_score: _Optional[float] = ..., min_translation_norm: _Optional[float] = ..., translation_blend_alpha: _Optional[float] = ..., translation_prior_increase: _Optional[float] = ...) -> None: ...
    class IrlsOutlierInitialization(_message.Message):
        __slots__ = ["activated", "cutoff", "rounds"]
        ACTIVATED_FIELD_NUMBER: _ClassVar[int]
        CUTOFF_FIELD_NUMBER: _ClassVar[int]
        ROUNDS_FIELD_NUMBER: _ClassVar[int]
        activated: bool
        cutoff: float
        rounds: int
        def __init__(self, activated: bool = ..., rounds: _Optional[int] = ..., cutoff: _Optional[float] = ...) -> None: ...
    class JointTrackEstimationOptions(_message.Message):
        __slots__ = ["motion_stride", "num_motion_models", "temporal_smoothing"]
        MOTION_STRIDE_FIELD_NUMBER: _ClassVar[int]
        NUM_MOTION_MODELS_FIELD_NUMBER: _ClassVar[int]
        TEMPORAL_SMOOTHING_FIELD_NUMBER: _ClassVar[int]
        motion_stride: int
        num_motion_models: int
        temporal_smoothing: bool
        def __init__(self, num_motion_models: _Optional[int] = ..., motion_stride: _Optional[int] = ..., temporal_smoothing: bool = ...) -> None: ...
    class LongFeatureBiasOptions(_message.Message):
        __slots__ = ["bias_stdev", "color_sigma", "grid_size", "inlier_bias", "inlier_irls_weight", "long_track_confidence_fraction", "long_track_threshold", "max_irls_change_ratio", "num_irls_observations", "outlier_bias", "seed_priors_from_bias", "spatial_sigma", "total_rounds", "use_spatial_bias"]
        BIAS_STDEV_FIELD_NUMBER: _ClassVar[int]
        COLOR_SIGMA_FIELD_NUMBER: _ClassVar[int]
        GRID_SIZE_FIELD_NUMBER: _ClassVar[int]
        INLIER_BIAS_FIELD_NUMBER: _ClassVar[int]
        INLIER_IRLS_WEIGHT_FIELD_NUMBER: _ClassVar[int]
        LONG_TRACK_CONFIDENCE_FRACTION_FIELD_NUMBER: _ClassVar[int]
        LONG_TRACK_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        MAX_IRLS_CHANGE_RATIO_FIELD_NUMBER: _ClassVar[int]
        NUM_IRLS_OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
        OUTLIER_BIAS_FIELD_NUMBER: _ClassVar[int]
        SEED_PRIORS_FROM_BIAS_FIELD_NUMBER: _ClassVar[int]
        SPATIAL_SIGMA_FIELD_NUMBER: _ClassVar[int]
        TOTAL_ROUNDS_FIELD_NUMBER: _ClassVar[int]
        USE_SPATIAL_BIAS_FIELD_NUMBER: _ClassVar[int]
        bias_stdev: float
        color_sigma: float
        grid_size: float
        inlier_bias: float
        inlier_irls_weight: float
        long_track_confidence_fraction: float
        long_track_threshold: int
        max_irls_change_ratio: float
        num_irls_observations: int
        outlier_bias: float
        seed_priors_from_bias: bool
        spatial_sigma: float
        total_rounds: int
        use_spatial_bias: bool
        def __init__(self, total_rounds: _Optional[int] = ..., inlier_bias: _Optional[float] = ..., outlier_bias: _Optional[float] = ..., num_irls_observations: _Optional[int] = ..., max_irls_change_ratio: _Optional[float] = ..., inlier_irls_weight: _Optional[float] = ..., bias_stdev: _Optional[float] = ..., use_spatial_bias: bool = ..., grid_size: _Optional[float] = ..., spatial_sigma: _Optional[float] = ..., color_sigma: _Optional[float] = ..., long_track_threshold: _Optional[int] = ..., long_track_confidence_fraction: _Optional[float] = ..., seed_priors_from_bias: bool = ...) -> None: ...
    class LongFeatureInitialization(_message.Message):
        __slots__ = ["activated", "min_length_percentile", "upweight_multiplier"]
        ACTIVATED_FIELD_NUMBER: _ClassVar[int]
        MIN_LENGTH_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
        UPWEIGHT_MULTIPLIER_FIELD_NUMBER: _ClassVar[int]
        activated: bool
        min_length_percentile: float
        upweight_multiplier: float
        def __init__(self, activated: bool = ..., min_length_percentile: _Optional[float] = ..., upweight_multiplier: _Optional[float] = ...) -> None: ...
    class MixtureHomographyBounds(_message.Message):
        __slots__ = ["frac_inlier_threshold", "max_adjacent_empty_blocks", "max_adjacent_outlier_blocks", "min_inlier_coverage"]
        FRAC_INLIER_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        MAX_ADJACENT_EMPTY_BLOCKS_FIELD_NUMBER: _ClassVar[int]
        MAX_ADJACENT_OUTLIER_BLOCKS_FIELD_NUMBER: _ClassVar[int]
        MIN_INLIER_COVERAGE_FIELD_NUMBER: _ClassVar[int]
        frac_inlier_threshold: float
        max_adjacent_empty_blocks: int
        max_adjacent_outlier_blocks: int
        min_inlier_coverage: float
        def __init__(self, min_inlier_coverage: _Optional[float] = ..., max_adjacent_outlier_blocks: _Optional[int] = ..., max_adjacent_empty_blocks: _Optional[int] = ..., frac_inlier_threshold: _Optional[float] = ...) -> None: ...
    class OverlayDetectionOptions(_message.Message):
        __slots__ = ["analysis_mask_size", "loose_near_zero_motion", "overlay_min_features", "overlay_min_ratio", "strict_max_translation_ratio", "strict_min_texturedness", "strict_near_zero_motion"]
        ANALYSIS_MASK_SIZE_FIELD_NUMBER: _ClassVar[int]
        LOOSE_NEAR_ZERO_MOTION_FIELD_NUMBER: _ClassVar[int]
        OVERLAY_MIN_FEATURES_FIELD_NUMBER: _ClassVar[int]
        OVERLAY_MIN_RATIO_FIELD_NUMBER: _ClassVar[int]
        STRICT_MAX_TRANSLATION_RATIO_FIELD_NUMBER: _ClassVar[int]
        STRICT_MIN_TEXTUREDNESS_FIELD_NUMBER: _ClassVar[int]
        STRICT_NEAR_ZERO_MOTION_FIELD_NUMBER: _ClassVar[int]
        analysis_mask_size: int
        loose_near_zero_motion: float
        overlay_min_features: float
        overlay_min_ratio: float
        strict_max_translation_ratio: float
        strict_min_texturedness: float
        strict_near_zero_motion: float
        def __init__(self, analysis_mask_size: _Optional[int] = ..., strict_near_zero_motion: _Optional[float] = ..., strict_max_translation_ratio: _Optional[float] = ..., strict_min_texturedness: _Optional[float] = ..., loose_near_zero_motion: _Optional[float] = ..., overlay_min_ratio: _Optional[float] = ..., overlay_min_features: _Optional[float] = ...) -> None: ...
    class ShotBoundaryOptions(_message.Message):
        __slots__ = ["appearance_consistency_threshold", "motion_consistency_threshold"]
        APPEARANCE_CONSISTENCY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        MOTION_CONSISTENCY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        appearance_consistency_threshold: float
        motion_consistency_threshold: float
        def __init__(self, motion_consistency_threshold: _Optional[float] = ..., appearance_consistency_threshold: _Optional[float] = ...) -> None: ...
    class SimilarityBounds(_message.Message):
        __slots__ = ["frac_inlier_threshold", "inlier_threshold", "limit_rotation", "lower_scale", "min_inlier_fraction", "min_inliers", "only_stable_input", "strict_inlier_threshold", "upper_scale"]
        FRAC_INLIER_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        INLIER_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        LIMIT_ROTATION_FIELD_NUMBER: _ClassVar[int]
        LOWER_SCALE_FIELD_NUMBER: _ClassVar[int]
        MIN_INLIERS_FIELD_NUMBER: _ClassVar[int]
        MIN_INLIER_FRACTION_FIELD_NUMBER: _ClassVar[int]
        ONLY_STABLE_INPUT_FIELD_NUMBER: _ClassVar[int]
        STRICT_INLIER_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        UPPER_SCALE_FIELD_NUMBER: _ClassVar[int]
        frac_inlier_threshold: float
        inlier_threshold: float
        limit_rotation: float
        lower_scale: float
        min_inlier_fraction: float
        min_inliers: float
        only_stable_input: bool
        strict_inlier_threshold: float
        upper_scale: float
        def __init__(self, only_stable_input: bool = ..., min_inlier_fraction: _Optional[float] = ..., min_inliers: _Optional[float] = ..., lower_scale: _Optional[float] = ..., upper_scale: _Optional[float] = ..., limit_rotation: _Optional[float] = ..., inlier_threshold: _Optional[float] = ..., frac_inlier_threshold: _Optional[float] = ..., strict_inlier_threshold: _Optional[float] = ...) -> None: ...
    class TranslationBounds(_message.Message):
        __slots__ = ["frac_max_motion_magnitude", "max_acceleration", "max_motion_stdev", "max_motion_stdev_threshold", "min_features"]
        FRAC_MAX_MOTION_MAGNITUDE_FIELD_NUMBER: _ClassVar[int]
        MAX_ACCELERATION_FIELD_NUMBER: _ClassVar[int]
        MAX_MOTION_STDEV_FIELD_NUMBER: _ClassVar[int]
        MAX_MOTION_STDEV_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
        MIN_FEATURES_FIELD_NUMBER: _ClassVar[int]
        frac_max_motion_magnitude: float
        max_acceleration: float
        max_motion_stdev: float
        max_motion_stdev_threshold: float
        min_features: int
        def __init__(self, min_features: _Optional[int] = ..., frac_max_motion_magnitude: _Optional[float] = ..., max_motion_stdev_threshold: _Optional[float] = ..., max_motion_stdev: _Optional[float] = ..., max_acceleration: _Optional[float] = ...) -> None: ...
    AFFINE_ESTIMATION_FIELD_NUMBER: _ClassVar[int]
    COVERAGE_GRID_SIZE_FIELD_NUMBER: _ClassVar[int]
    DEACTIVATE_STABLE_MOTION_ESTIMATION_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_LIMITED_IRLS_SCALING_FIELD_NUMBER: _ClassVar[int]
    ESTIMATE_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    ESTIMATE_TRANSLATION_IRLS_FIELD_NUMBER: _ClassVar[int]
    ESTIMATION_AFFINE_IRLS: MotionEstimationOptions.AffineEstimation
    ESTIMATION_AFFINE_L2: MotionEstimationOptions.AffineEstimation
    ESTIMATION_AFFINE_NONE: MotionEstimationOptions.AffineEstimation
    ESTIMATION_HOMOG_IRLS: MotionEstimationOptions.HomographyEstimation
    ESTIMATION_HOMOG_L2: MotionEstimationOptions.HomographyEstimation
    ESTIMATION_HOMOG_MIX_IRLS: MotionEstimationOptions.MixtureHomographyEstimation
    ESTIMATION_HOMOG_MIX_L2: MotionEstimationOptions.MixtureHomographyEstimation
    ESTIMATION_HOMOG_MIX_NONE: MotionEstimationOptions.MixtureHomographyEstimation
    ESTIMATION_HOMOG_NONE: MotionEstimationOptions.HomographyEstimation
    ESTIMATION_LS_IRLS: MotionEstimationOptions.LinearSimilarityEstimation
    ESTIMATION_LS_L1: MotionEstimationOptions.LinearSimilarityEstimation
    ESTIMATION_LS_L2: MotionEstimationOptions.LinearSimilarityEstimation
    ESTIMATION_LS_L2_RANSAC: MotionEstimationOptions.LinearSimilarityEstimation
    ESTIMATION_LS_NONE: MotionEstimationOptions.LinearSimilarityEstimation
    ESTIMATION_POLICY_FIELD_NUMBER: _ClassVar[int]
    Extensions: _python_message._ExtensionDict
    FEATURE_DENSITY_NORMALIZATION_FIELD_NUMBER: _ClassVar[int]
    FEATURE_GRID_SIZE_FIELD_NUMBER: _ClassVar[int]
    FEATURE_MASK_SIZE_FIELD_NUMBER: _ClassVar[int]
    FEATURE_SIGMA_FIELD_NUMBER: _ClassVar[int]
    FILTER_5_TAPS_FIELD_NUMBER: _ClassVar[int]
    FILTER_INITIALIZED_IRLS_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    FRAME_CONFIDENCE_WEIGHTING_FIELD_NUMBER: _ClassVar[int]
    FULL_MIXTURE: MotionEstimationOptions.MixtureModelMode
    HOMOGRAPHY_ESTIMATION_FIELD_NUMBER: _ClassVar[int]
    HOMOGRAPHY_EXACT_DENOMINATOR_SCALING_FIELD_NUMBER: _ClassVar[int]
    HOMOGRAPHY_IRLS_WEIGHT_INITIALIZATION_FIELD_NUMBER: _ClassVar[int]
    HOMOGRAPHY_PERSPECTIVE_REGULARIZER_FIELD_NUMBER: _ClassVar[int]
    INDEPENDENT_PARALLEL: MotionEstimationOptions.EstimationPolicy
    IRLS_FILTER_CORNER_RESPONSE: MotionEstimationOptions.IRLSWeightFilter
    IRLS_FILTER_NONE: MotionEstimationOptions.IRLSWeightFilter
    IRLS_FILTER_TEXTURE: MotionEstimationOptions.IRLSWeightFilter
    IRLS_INITIALIZATION_FIELD_NUMBER: _ClassVar[int]
    IRLS_MASK_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    IRLS_MIXTURE_FRACTION_SCALE_FIELD_NUMBER: _ClassVar[int]
    IRLS_MOTION_MAGNITUDE_FRACTION_FIELD_NUMBER: _ClassVar[int]
    IRLS_PRIOR_SCALE_FIELD_NUMBER: _ClassVar[int]
    IRLS_ROUNDS_FIELD_NUMBER: _ClassVar[int]
    IRLS_USE_L0_NORM_FIELD_NUMBER: _ClassVar[int]
    IRLS_WEIGHTS_PREINITIALIZED_FIELD_NUMBER: _ClassVar[int]
    IRLS_WEIGHT_CENTER_GAUSSIAN: MotionEstimationOptions.HomographyIrlsWeightInitialization
    IRLS_WEIGHT_CONSTANT_ONE: MotionEstimationOptions.HomographyIrlsWeightInitialization
    IRLS_WEIGHT_FILTER_FIELD_NUMBER: _ClassVar[int]
    IRLS_WEIGHT_PERIMETER_GAUSSIAN: MotionEstimationOptions.HomographyIrlsWeightInitialization
    IRLS_WEIGHT_UNKNOWN: MotionEstimationOptions.HomographyIrlsWeightInitialization
    JOINTLY_FROM_TRACKS: MotionEstimationOptions.EstimationPolicy
    JOINT_TRACK_ESTIMATION_FIELD_NUMBER: _ClassVar[int]
    LABEL_EMPTY_FRAMES_AS_VALID_FIELD_NUMBER: _ClassVar[int]
    LINEAR_SIMILARITY_ESTIMATION_FIELD_NUMBER: _ClassVar[int]
    LIN_SIM_INLIER_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    LONG_FEATURE_BIAS_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    LONG_FEATURE_INITIALIZATION_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_MODEL_MODE_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_REGULARIZER_BASE_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_REGULARIZER_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_REGULARIZER_LEVELS_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_ROW_SIGMA_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_RS_ANALYSIS_LEVEL_FIELD_NUMBER: _ClassVar[int]
    MIX_HOMOGRAPHY_ESTIMATION_FIELD_NUMBER: _ClassVar[int]
    NUM_MIXTURES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_REFINED_IRLS_WEIGHTS_FIELD_NUMBER: _ClassVar[int]
    OVERLAY_ANALYSIS_CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    OVERLAY_DETECTION_FIELD_NUMBER: _ClassVar[int]
    OVERLAY_DETECTION_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    PROJECT_VALID_MOTIONS_DOWN_FIELD_NUMBER: _ClassVar[int]
    RESET_CONFIDENCE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SHOT_BOUNDARY_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    SKEW_ROTATION_MIXTURE: MotionEstimationOptions.MixtureModelMode
    SPATIAL_SIGMA_FIELD_NUMBER: _ClassVar[int]
    STABLE_HOMOGRAPHY_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    STABLE_MIXTURE_HOMOGRAPHY_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    STABLE_SIMILARITY_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    STABLE_TRANSLATION_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    STRICT_COVERAGE_SCALE_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_IRLS_DIAMETER_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_IRLS_MASK: MotionEstimationOptions.EstimationPolicy
    TEMPORAL_LONG_FEATURE_BIAS: MotionEstimationOptions.EstimationPolicy
    TEMPORAL_SIGMA_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_MIXTURE: MotionEstimationOptions.MixtureModelMode
    UNKNOWN: MotionEstimationOptions.EstimationPolicy
    USE_EXACT_HOMOGRAPHY_ESTIMATION_FIELD_NUMBER: _ClassVar[int]
    USE_HIGHEST_ACCURACY_FOR_NORMAL_EQUATIONS_FIELD_NUMBER: _ClassVar[int]
    USE_ONLY_LIN_SIM_INLIERS_FOR_HOMOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    affine_estimation: MotionEstimationOptions.AffineEstimation
    coverage_grid_size: int
    deactivate_stable_motion_estimation: bool
    domain_limited_irls_scaling: bool
    estimate_similarity: bool
    estimate_translation_irls: bool
    estimation_policy: MotionEstimationOptions.EstimationPolicy
    feature_density_normalization: bool
    feature_grid_size: float
    feature_mask_size: int
    feature_sigma: float
    filter_5_taps: bool
    filter_initialized_irls_weights: bool
    frame_confidence_weighting: bool
    homography_estimation: MotionEstimationOptions.HomographyEstimation
    homography_exact_denominator_scaling: bool
    homography_irls_weight_initialization: MotionEstimationOptions.HomographyIrlsWeightInitialization
    homography_perspective_regularizer: float
    irls_initialization: MotionEstimationOptions.IrlsOutlierInitialization
    irls_mask_options: MotionEstimationOptions.IrlsMaskOptions
    irls_mixture_fraction_scale: float
    irls_motion_magnitude_fraction: float
    irls_prior_scale: float
    irls_rounds: int
    irls_use_l0_norm: bool
    irls_weight_filter: MotionEstimationOptions.IRLSWeightFilter
    irls_weights_preinitialized: bool
    joint_track_estimation: MotionEstimationOptions.JointTrackEstimationOptions
    label_empty_frames_as_valid: bool
    lin_sim_inlier_threshold: float
    linear_similarity_estimation: MotionEstimationOptions.LinearSimilarityEstimation
    long_feature_bias_options: MotionEstimationOptions.LongFeatureBiasOptions
    long_feature_initialization: MotionEstimationOptions.LongFeatureInitialization
    mix_homography_estimation: MotionEstimationOptions.MixtureHomographyEstimation
    mixture_model_mode: MotionEstimationOptions.MixtureModelMode
    mixture_regularizer: float
    mixture_regularizer_base: float
    mixture_regularizer_levels: float
    mixture_row_sigma: float
    mixture_rs_analysis_level: int
    num_mixtures: int
    output_refined_irls_weights: bool
    overlay_analysis_chunk_size: int
    overlay_detection: bool
    overlay_detection_options: MotionEstimationOptions.OverlayDetectionOptions
    project_valid_motions_down: bool
    reset_confidence_threshold: float
    shot_boundary_options: MotionEstimationOptions.ShotBoundaryOptions
    spatial_sigma: float
    stable_homography_bounds: MotionEstimationOptions.HomographyBounds
    stable_mixture_homography_bounds: MotionEstimationOptions.MixtureHomographyBounds
    stable_similarity_bounds: MotionEstimationOptions.SimilarityBounds
    stable_translation_bounds: MotionEstimationOptions.TranslationBounds
    strict_coverage_scale: float
    temporal_irls_diameter: int
    temporal_sigma: float
    use_exact_homography_estimation: bool
    use_highest_accuracy_for_normal_equations: bool
    use_only_lin_sim_inliers_for_homography: bool
    def __init__(self, estimate_translation_irls: bool = ..., linear_similarity_estimation: _Optional[_Union[MotionEstimationOptions.LinearSimilarityEstimation, str]] = ..., affine_estimation: _Optional[_Union[MotionEstimationOptions.AffineEstimation, str]] = ..., homography_estimation: _Optional[_Union[MotionEstimationOptions.HomographyEstimation, str]] = ..., homography_exact_denominator_scaling: bool = ..., use_exact_homography_estimation: bool = ..., use_highest_accuracy_for_normal_equations: bool = ..., homography_perspective_regularizer: _Optional[float] = ..., mix_homography_estimation: _Optional[_Union[MotionEstimationOptions.MixtureHomographyEstimation, str]] = ..., num_mixtures: _Optional[int] = ..., mixture_row_sigma: _Optional[float] = ..., mixture_regularizer: _Optional[float] = ..., mixture_regularizer_levels: _Optional[float] = ..., mixture_regularizer_base: _Optional[float] = ..., mixture_rs_analysis_level: _Optional[int] = ..., irls_rounds: _Optional[int] = ..., irls_prior_scale: _Optional[float] = ..., irls_motion_magnitude_fraction: _Optional[float] = ..., irls_mixture_fraction_scale: _Optional[float] = ..., irls_weights_preinitialized: bool = ..., filter_initialized_irls_weights: bool = ..., irls_initialization: _Optional[_Union[MotionEstimationOptions.IrlsOutlierInitialization, _Mapping]] = ..., feature_density_normalization: bool = ..., feature_mask_size: _Optional[int] = ..., long_feature_initialization: _Optional[_Union[MotionEstimationOptions.LongFeatureInitialization, _Mapping]] = ..., irls_mask_options: _Optional[_Union[MotionEstimationOptions.IrlsMaskOptions, _Mapping]] = ..., joint_track_estimation: _Optional[_Union[MotionEstimationOptions.JointTrackEstimationOptions, _Mapping]] = ..., long_feature_bias_options: _Optional[_Union[MotionEstimationOptions.LongFeatureBiasOptions, _Mapping]] = ..., estimation_policy: _Optional[_Union[MotionEstimationOptions.EstimationPolicy, str]] = ..., coverage_grid_size: _Optional[int] = ..., mixture_model_mode: _Optional[_Union[MotionEstimationOptions.MixtureModelMode, str]] = ..., use_only_lin_sim_inliers_for_homography: bool = ..., lin_sim_inlier_threshold: _Optional[float] = ..., stable_translation_bounds: _Optional[_Union[MotionEstimationOptions.TranslationBounds, _Mapping]] = ..., stable_similarity_bounds: _Optional[_Union[MotionEstimationOptions.SimilarityBounds, _Mapping]] = ..., stable_homography_bounds: _Optional[_Union[MotionEstimationOptions.HomographyBounds, _Mapping]] = ..., stable_mixture_homography_bounds: _Optional[_Union[MotionEstimationOptions.MixtureHomographyBounds, _Mapping]] = ..., strict_coverage_scale: _Optional[float] = ..., label_empty_frames_as_valid: bool = ..., feature_grid_size: _Optional[float] = ..., spatial_sigma: _Optional[float] = ..., temporal_irls_diameter: _Optional[int] = ..., temporal_sigma: _Optional[float] = ..., feature_sigma: _Optional[float] = ..., filter_5_taps: bool = ..., frame_confidence_weighting: bool = ..., reset_confidence_threshold: _Optional[float] = ..., irls_weight_filter: _Optional[_Union[MotionEstimationOptions.IRLSWeightFilter, str]] = ..., overlay_detection: bool = ..., overlay_analysis_chunk_size: _Optional[int] = ..., overlay_detection_options: _Optional[_Union[MotionEstimationOptions.OverlayDetectionOptions, _Mapping]] = ..., shot_boundary_options: _Optional[_Union[MotionEstimationOptions.ShotBoundaryOptions, _Mapping]] = ..., output_refined_irls_weights: bool = ..., homography_irls_weight_initialization: _Optional[_Union[MotionEstimationOptions.HomographyIrlsWeightInitialization, str]] = ..., irls_use_l0_norm: bool = ..., domain_limited_irls_scaling: bool = ..., deactivate_stable_motion_estimation: bool = ..., project_valid_motions_down: bool = ..., estimate_similarity: bool = ...) -> None: ...
