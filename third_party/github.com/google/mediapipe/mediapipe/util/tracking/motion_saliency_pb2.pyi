from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class MotionSaliencyOptions(_message.Message):
    __slots__ = ["bound_bottom", "bound_left", "bound_right", "bound_top", "filtering_sigma_space", "filtering_sigma_time", "min_features", "min_irls_mode_weight", "mode_band_width", "num_top_irls_modes", "saliency_weight", "scale_weight_by_flow_magnitude", "selection_frame_radius", "selection_minimum_support", "selection_support_distance", "use_only_foreground_regions"]
    BOUND_BOTTOM_FIELD_NUMBER: _ClassVar[int]
    BOUND_LEFT_FIELD_NUMBER: _ClassVar[int]
    BOUND_RIGHT_FIELD_NUMBER: _ClassVar[int]
    BOUND_TOP_FIELD_NUMBER: _ClassVar[int]
    FILTERING_SIGMA_SPACE_FIELD_NUMBER: _ClassVar[int]
    FILTERING_SIGMA_TIME_FIELD_NUMBER: _ClassVar[int]
    MIN_FEATURES_FIELD_NUMBER: _ClassVar[int]
    MIN_IRLS_MODE_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    MODE_BAND_WIDTH_FIELD_NUMBER: _ClassVar[int]
    NUM_TOP_IRLS_MODES_FIELD_NUMBER: _ClassVar[int]
    SALIENCY_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    SCALE_WEIGHT_BY_FLOW_MAGNITUDE_FIELD_NUMBER: _ClassVar[int]
    SELECTION_FRAME_RADIUS_FIELD_NUMBER: _ClassVar[int]
    SELECTION_MINIMUM_SUPPORT_FIELD_NUMBER: _ClassVar[int]
    SELECTION_SUPPORT_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    USE_ONLY_FOREGROUND_REGIONS_FIELD_NUMBER: _ClassVar[int]
    bound_bottom: float
    bound_left: float
    bound_right: float
    bound_top: float
    filtering_sigma_space: float
    filtering_sigma_time: float
    min_features: int
    min_irls_mode_weight: float
    mode_band_width: float
    num_top_irls_modes: int
    saliency_weight: float
    scale_weight_by_flow_magnitude: bool
    selection_frame_radius: int
    selection_minimum_support: int
    selection_support_distance: float
    use_only_foreground_regions: bool
    def __init__(self, bound_left: _Optional[float] = ..., bound_bottom: _Optional[float] = ..., bound_right: _Optional[float] = ..., bound_top: _Optional[float] = ..., saliency_weight: _Optional[float] = ..., scale_weight_by_flow_magnitude: bool = ..., min_features: _Optional[int] = ..., use_only_foreground_regions: bool = ..., min_irls_mode_weight: _Optional[float] = ..., num_top_irls_modes: _Optional[int] = ..., mode_band_width: _Optional[float] = ..., selection_frame_radius: _Optional[int] = ..., selection_support_distance: _Optional[float] = ..., selection_minimum_support: _Optional[int] = ..., filtering_sigma_space: _Optional[float] = ..., filtering_sigma_time: _Optional[float] = ...) -> None: ...
