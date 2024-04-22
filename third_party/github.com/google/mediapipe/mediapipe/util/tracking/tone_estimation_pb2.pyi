from mediapipe.util.tracking import tone_models_pb2 as _tone_models_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ClipMaskOptions(_message.Message):
    __slots__ = ["clip_mask_diameter", "max_clipped_channels", "max_exposure", "min_exposure"]
    CLIP_MASK_DIAMETER_FIELD_NUMBER: _ClassVar[int]
    MAX_CLIPPED_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    MAX_EXPOSURE_FIELD_NUMBER: _ClassVar[int]
    MIN_EXPOSURE_FIELD_NUMBER: _ClassVar[int]
    clip_mask_diameter: int
    max_clipped_channels: int
    max_exposure: float
    min_exposure: float
    def __init__(self, min_exposure: _Optional[float] = ..., max_exposure: _Optional[float] = ..., max_clipped_channels: _Optional[int] = ..., clip_mask_diameter: _Optional[int] = ...) -> None: ...

class PatchToneMatch(_message.Message):
    __slots__ = ["irls_weight", "tone_match"]
    IRLS_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    TONE_MATCH_FIELD_NUMBER: _ClassVar[int]
    irls_weight: float
    tone_match: _containers.RepeatedCompositeFieldContainer[ToneMatch]
    def __init__(self, tone_match: _Optional[_Iterable[_Union[ToneMatch, _Mapping]]] = ..., irls_weight: _Optional[float] = ...) -> None: ...

class ToneChange(_message.Message):
    __slots__ = ["affine", "frac_clipped", "gain_bias", "high_mid_percentile", "high_percentile", "log_domain", "low_mid_percentile", "low_percentile", "mid_percentile", "mixture_affine", "mixture_domain_sigma", "mixture_gain_bias", "stability_stats", "type"]
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class StabilityStats(_message.Message):
        __slots__ = ["inlier_fraction", "inlier_weight", "num_inliers"]
        INLIER_FRACTION_FIELD_NUMBER: _ClassVar[int]
        INLIER_WEIGHT_FIELD_NUMBER: _ClassVar[int]
        NUM_INLIERS_FIELD_NUMBER: _ClassVar[int]
        inlier_fraction: float
        inlier_weight: float
        num_inliers: int
        def __init__(self, num_inliers: _Optional[int] = ..., inlier_fraction: _Optional[float] = ..., inlier_weight: _Optional[float] = ...) -> None: ...
    AFFINE_FIELD_NUMBER: _ClassVar[int]
    FRAC_CLIPPED_FIELD_NUMBER: _ClassVar[int]
    GAIN_BIAS_FIELD_NUMBER: _ClassVar[int]
    HIGH_MID_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    HIGH_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    LOG_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    LOW_MID_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    LOW_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    MID_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_AFFINE_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_DOMAIN_SIGMA_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_GAIN_BIAS_FIELD_NUMBER: _ClassVar[int]
    STABILITY_STATS_FIELD_NUMBER: _ClassVar[int]
    TONE_TYPE_INVALID: ToneChange.Type
    TONE_TYPE_VALID: ToneChange.Type
    TYPE_FIELD_NUMBER: _ClassVar[int]
    affine: _tone_models_pb2.AffineToneModel
    frac_clipped: float
    gain_bias: _tone_models_pb2.GainBiasModel
    high_mid_percentile: float
    high_percentile: float
    log_domain: bool
    low_mid_percentile: float
    low_percentile: float
    mid_percentile: float
    mixture_affine: _tone_models_pb2.MixtureAffineToneModel
    mixture_domain_sigma: float
    mixture_gain_bias: _tone_models_pb2.MixtureGainBiasModel
    stability_stats: ToneChange.StabilityStats
    type: ToneChange.Type
    def __init__(self, gain_bias: _Optional[_Union[_tone_models_pb2.GainBiasModel, _Mapping]] = ..., affine: _Optional[_Union[_tone_models_pb2.AffineToneModel, _Mapping]] = ..., mixture_gain_bias: _Optional[_Union[_tone_models_pb2.MixtureGainBiasModel, _Mapping]] = ..., mixture_affine: _Optional[_Union[_tone_models_pb2.MixtureAffineToneModel, _Mapping]] = ..., mixture_domain_sigma: _Optional[float] = ..., frac_clipped: _Optional[float] = ..., low_percentile: _Optional[float] = ..., low_mid_percentile: _Optional[float] = ..., mid_percentile: _Optional[float] = ..., high_mid_percentile: _Optional[float] = ..., high_percentile: _Optional[float] = ..., log_domain: bool = ..., type: _Optional[_Union[ToneChange.Type, str]] = ..., stability_stats: _Optional[_Union[ToneChange.StabilityStats, _Mapping]] = ...) -> None: ...

class ToneEstimationOptions(_message.Message):
    __slots__ = ["clip_mask_options", "downsample_factor", "downsample_mode", "downsampling_size", "irls_iterations", "stable_gain_bias_bounds", "stats_high_mid_percentile", "stats_high_percentile", "stats_low_mid_percentile", "stats_low_percentile", "stats_mid_percentile", "tone_match_options"]
    class DownsampleMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class GainBiasBounds(_message.Message):
        __slots__ = ["lower_bias", "lower_gain", "min_inlier_fraction", "min_inlier_weight", "upper_bias", "upper_gain"]
        LOWER_BIAS_FIELD_NUMBER: _ClassVar[int]
        LOWER_GAIN_FIELD_NUMBER: _ClassVar[int]
        MIN_INLIER_FRACTION_FIELD_NUMBER: _ClassVar[int]
        MIN_INLIER_WEIGHT_FIELD_NUMBER: _ClassVar[int]
        UPPER_BIAS_FIELD_NUMBER: _ClassVar[int]
        UPPER_GAIN_FIELD_NUMBER: _ClassVar[int]
        lower_bias: float
        lower_gain: float
        min_inlier_fraction: float
        min_inlier_weight: float
        upper_bias: float
        upper_gain: float
        def __init__(self, min_inlier_fraction: _Optional[float] = ..., min_inlier_weight: _Optional[float] = ..., lower_gain: _Optional[float] = ..., upper_gain: _Optional[float] = ..., lower_bias: _Optional[float] = ..., upper_bias: _Optional[float] = ...) -> None: ...
    CLIP_MASK_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    DOWNSAMPLE_FACTOR_FIELD_NUMBER: _ClassVar[int]
    DOWNSAMPLE_MODE_FIELD_NUMBER: _ClassVar[int]
    DOWNSAMPLING_SIZE_FIELD_NUMBER: _ClassVar[int]
    IRLS_ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    STABLE_GAIN_BIAS_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    STATS_HIGH_MID_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    STATS_HIGH_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    STATS_LOW_MID_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    STATS_LOW_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    STATS_MID_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    TONE_DOWNSAMPLE_BY_FACTOR: ToneEstimationOptions.DownsampleMode
    TONE_DOWNSAMPLE_NONE: ToneEstimationOptions.DownsampleMode
    TONE_DOWNSAMPLE_TO_MAX_SIZE: ToneEstimationOptions.DownsampleMode
    TONE_DOWNSAMPLE_TO_MIN_SIZE: ToneEstimationOptions.DownsampleMode
    TONE_DOWNSAMPLE_UNKNOWN: ToneEstimationOptions.DownsampleMode
    TONE_MATCH_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    clip_mask_options: ClipMaskOptions
    downsample_factor: float
    downsample_mode: ToneEstimationOptions.DownsampleMode
    downsampling_size: int
    irls_iterations: int
    stable_gain_bias_bounds: ToneEstimationOptions.GainBiasBounds
    stats_high_mid_percentile: float
    stats_high_percentile: float
    stats_low_mid_percentile: float
    stats_low_percentile: float
    stats_mid_percentile: float
    tone_match_options: ToneMatchOptions
    def __init__(self, tone_match_options: _Optional[_Union[ToneMatchOptions, _Mapping]] = ..., clip_mask_options: _Optional[_Union[ClipMaskOptions, _Mapping]] = ..., stats_low_percentile: _Optional[float] = ..., stats_low_mid_percentile: _Optional[float] = ..., stats_mid_percentile: _Optional[float] = ..., stats_high_mid_percentile: _Optional[float] = ..., stats_high_percentile: _Optional[float] = ..., irls_iterations: _Optional[int] = ..., stable_gain_bias_bounds: _Optional[_Union[ToneEstimationOptions.GainBiasBounds, _Mapping]] = ..., downsample_mode: _Optional[_Union[ToneEstimationOptions.DownsampleMode, str]] = ..., downsampling_size: _Optional[int] = ..., downsample_factor: _Optional[float] = ...) -> None: ...

class ToneMatch(_message.Message):
    __slots__ = ["curr_val", "prev_val"]
    CURR_VAL_FIELD_NUMBER: _ClassVar[int]
    PREV_VAL_FIELD_NUMBER: _ClassVar[int]
    curr_val: float
    prev_val: float
    def __init__(self, curr_val: _Optional[float] = ..., prev_val: _Optional[float] = ...) -> None: ...

class ToneMatchOptions(_message.Message):
    __slots__ = ["log_domain", "match_percentile_steps", "max_frac_clipped", "max_match_percentile", "min_match_percentile", "patch_radius"]
    LOG_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    MATCH_PERCENTILE_STEPS_FIELD_NUMBER: _ClassVar[int]
    MAX_FRAC_CLIPPED_FIELD_NUMBER: _ClassVar[int]
    MAX_MATCH_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    MIN_MATCH_PERCENTILE_FIELD_NUMBER: _ClassVar[int]
    PATCH_RADIUS_FIELD_NUMBER: _ClassVar[int]
    log_domain: bool
    match_percentile_steps: int
    max_frac_clipped: float
    max_match_percentile: float
    min_match_percentile: float
    patch_radius: int
    def __init__(self, min_match_percentile: _Optional[float] = ..., max_match_percentile: _Optional[float] = ..., match_percentile_steps: _Optional[int] = ..., patch_radius: _Optional[int] = ..., max_frac_clipped: _Optional[float] = ..., log_domain: bool = ...) -> None: ...
