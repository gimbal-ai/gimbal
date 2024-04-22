from mediapipe.util.tracking import motion_models_pb2 as _motion_models_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MotionBoxInternalState(_message.Message):
    __slots__ = ["camera_dx", "camera_dy", "dx", "dy", "inlier_score", "pos_x", "pos_y", "track_id"]
    CAMERA_DX_FIELD_NUMBER: _ClassVar[int]
    CAMERA_DY_FIELD_NUMBER: _ClassVar[int]
    DX_FIELD_NUMBER: _ClassVar[int]
    DY_FIELD_NUMBER: _ClassVar[int]
    INLIER_SCORE_FIELD_NUMBER: _ClassVar[int]
    POS_X_FIELD_NUMBER: _ClassVar[int]
    POS_Y_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    camera_dx: _containers.RepeatedScalarFieldContainer[float]
    camera_dy: _containers.RepeatedScalarFieldContainer[float]
    dx: _containers.RepeatedScalarFieldContainer[float]
    dy: _containers.RepeatedScalarFieldContainer[float]
    inlier_score: _containers.RepeatedScalarFieldContainer[float]
    pos_x: _containers.RepeatedScalarFieldContainer[float]
    pos_y: _containers.RepeatedScalarFieldContainer[float]
    track_id: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, pos_x: _Optional[_Iterable[float]] = ..., pos_y: _Optional[_Iterable[float]] = ..., dx: _Optional[_Iterable[float]] = ..., dy: _Optional[_Iterable[float]] = ..., camera_dx: _Optional[_Iterable[float]] = ..., camera_dy: _Optional[_Iterable[float]] = ..., track_id: _Optional[_Iterable[int]] = ..., inlier_score: _Optional[_Iterable[float]] = ...) -> None: ...

class MotionBoxState(_message.Message):
    __slots__ = ["aspect_ratio", "background_discrimination", "dx", "dy", "height", "inlier_center_x", "inlier_center_y", "inlier_height", "inlier_id_match_pos", "inlier_ids", "inlier_length", "inlier_ratio", "inlier_sum", "inlier_width", "internal", "kinetic_energy", "motion_disparity", "outlier_id_match_pos", "outlier_ids", "pnp_homography", "pos_x", "pos_y", "prior_diff", "prior_weight", "quad", "request_grouping", "rotation", "scale", "spatial_confidence", "spatial_prior", "spatial_prior_grid_size", "track_status", "tracking_confidence", "width"]
    class TrackStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class Quad(_message.Message):
        __slots__ = ["vertices"]
        VERTICES_FIELD_NUMBER: _ClassVar[int]
        vertices: _containers.RepeatedScalarFieldContainer[float]
        def __init__(self, vertices: _Optional[_Iterable[float]] = ...) -> None: ...
    ASPECT_RATIO_FIELD_NUMBER: _ClassVar[int]
    BACKGROUND_DISCRIMINATION_FIELD_NUMBER: _ClassVar[int]
    BOX_DUPLICATED: MotionBoxState.TrackStatus
    BOX_EMPTY: MotionBoxState.TrackStatus
    BOX_NO_FEATURES: MotionBoxState.TrackStatus
    BOX_TRACKED: MotionBoxState.TrackStatus
    BOX_TRACKED_OUT_OF_BOUND: MotionBoxState.TrackStatus
    BOX_UNTRACKED: MotionBoxState.TrackStatus
    DX_FIELD_NUMBER: _ClassVar[int]
    DY_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    INLIER_CENTER_X_FIELD_NUMBER: _ClassVar[int]
    INLIER_CENTER_Y_FIELD_NUMBER: _ClassVar[int]
    INLIER_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    INLIER_IDS_FIELD_NUMBER: _ClassVar[int]
    INLIER_ID_MATCH_POS_FIELD_NUMBER: _ClassVar[int]
    INLIER_LENGTH_FIELD_NUMBER: _ClassVar[int]
    INLIER_RATIO_FIELD_NUMBER: _ClassVar[int]
    INLIER_SUM_FIELD_NUMBER: _ClassVar[int]
    INLIER_WIDTH_FIELD_NUMBER: _ClassVar[int]
    INTERNAL_FIELD_NUMBER: _ClassVar[int]
    KINETIC_ENERGY_FIELD_NUMBER: _ClassVar[int]
    MOTION_DISPARITY_FIELD_NUMBER: _ClassVar[int]
    OUTLIER_IDS_FIELD_NUMBER: _ClassVar[int]
    OUTLIER_ID_MATCH_POS_FIELD_NUMBER: _ClassVar[int]
    PNP_HOMOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    POS_X_FIELD_NUMBER: _ClassVar[int]
    POS_Y_FIELD_NUMBER: _ClassVar[int]
    PRIOR_DIFF_FIELD_NUMBER: _ClassVar[int]
    PRIOR_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    QUAD_FIELD_NUMBER: _ClassVar[int]
    REQUEST_GROUPING_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_PRIOR_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_PRIOR_GRID_SIZE_FIELD_NUMBER: _ClassVar[int]
    TRACKING_CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    TRACK_STATUS_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    aspect_ratio: float
    background_discrimination: float
    dx: float
    dy: float
    height: float
    inlier_center_x: float
    inlier_center_y: float
    inlier_height: float
    inlier_id_match_pos: _containers.RepeatedScalarFieldContainer[int]
    inlier_ids: _containers.RepeatedScalarFieldContainer[int]
    inlier_length: _containers.RepeatedScalarFieldContainer[int]
    inlier_ratio: float
    inlier_sum: float
    inlier_width: float
    internal: MotionBoxInternalState
    kinetic_energy: float
    motion_disparity: float
    outlier_id_match_pos: _containers.RepeatedScalarFieldContainer[int]
    outlier_ids: _containers.RepeatedScalarFieldContainer[int]
    pnp_homography: _motion_models_pb2.Homography
    pos_x: float
    pos_y: float
    prior_diff: float
    prior_weight: float
    quad: MotionBoxState.Quad
    request_grouping: bool
    rotation: float
    scale: float
    spatial_confidence: _containers.RepeatedScalarFieldContainer[float]
    spatial_prior: _containers.RepeatedScalarFieldContainer[float]
    spatial_prior_grid_size: int
    track_status: MotionBoxState.TrackStatus
    tracking_confidence: float
    width: float
    def __init__(self, pos_x: _Optional[float] = ..., pos_y: _Optional[float] = ..., width: _Optional[float] = ..., height: _Optional[float] = ..., scale: _Optional[float] = ..., rotation: _Optional[float] = ..., quad: _Optional[_Union[MotionBoxState.Quad, _Mapping]] = ..., aspect_ratio: _Optional[float] = ..., request_grouping: bool = ..., pnp_homography: _Optional[_Union[_motion_models_pb2.Homography, _Mapping]] = ..., dx: _Optional[float] = ..., dy: _Optional[float] = ..., kinetic_energy: _Optional[float] = ..., prior_weight: _Optional[float] = ..., track_status: _Optional[_Union[MotionBoxState.TrackStatus, str]] = ..., spatial_prior_grid_size: _Optional[int] = ..., spatial_prior: _Optional[_Iterable[float]] = ..., spatial_confidence: _Optional[_Iterable[float]] = ..., prior_diff: _Optional[float] = ..., motion_disparity: _Optional[float] = ..., background_discrimination: _Optional[float] = ..., inlier_center_x: _Optional[float] = ..., inlier_center_y: _Optional[float] = ..., inlier_sum: _Optional[float] = ..., inlier_ratio: _Optional[float] = ..., inlier_width: _Optional[float] = ..., inlier_height: _Optional[float] = ..., inlier_ids: _Optional[_Iterable[int]] = ..., inlier_id_match_pos: _Optional[_Iterable[int]] = ..., inlier_length: _Optional[_Iterable[int]] = ..., outlier_ids: _Optional[_Iterable[int]] = ..., outlier_id_match_pos: _Optional[_Iterable[int]] = ..., tracking_confidence: _Optional[float] = ..., internal: _Optional[_Union[MotionBoxInternalState, _Mapping]] = ...) -> None: ...

class TrackStepOptions(_message.Message):
    __slots__ = ["background_discrimination_high_level", "background_discrimination_low_level", "box_similarity_max_rotation", "box_similarity_max_scale", "camera_intrinsics", "cancel_tracking_with_occlusion_options", "compute_spatial_prior", "disparity_decay", "expansion_size", "forced_pnp_tracking", "high_kinetic_energy", "inlier_center_relative_distance", "inlier_high_weight", "inlier_low_weight", "inlier_spring_force", "irls_initialization", "irls_iterations", "kinetic_center_relative_distance", "kinetic_energy_decay", "kinetic_spring_force", "kinetic_spring_force_min_kinetic_energy", "low_kinetic_energy", "max_track_failures", "min_motion_sigma", "motion_disparity_high_level", "motion_disparity_low_level", "motion_prior_weight", "object_similarity_min_contd_inliers", "prior_weight_increase", "quad_homography_max_rotation", "quad_homography_max_scale", "relative_motion_sigma", "return_internal_state", "spatial_sigma", "static_motion_temporal_ratio", "track_object_and_camera", "tracking_degrees", "use_post_estimation_weights_for_state", "velocity_update_weight"]
    class TrackingDegrees(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class CameraIntrinsics(_message.Message):
        __slots__ = ["cx", "cy", "fx", "fy", "h", "k0", "k1", "k2", "w"]
        CX_FIELD_NUMBER: _ClassVar[int]
        CY_FIELD_NUMBER: _ClassVar[int]
        FX_FIELD_NUMBER: _ClassVar[int]
        FY_FIELD_NUMBER: _ClassVar[int]
        H_FIELD_NUMBER: _ClassVar[int]
        K0_FIELD_NUMBER: _ClassVar[int]
        K1_FIELD_NUMBER: _ClassVar[int]
        K2_FIELD_NUMBER: _ClassVar[int]
        W_FIELD_NUMBER: _ClassVar[int]
        cx: float
        cy: float
        fx: float
        fy: float
        h: int
        k0: float
        k1: float
        k2: float
        w: int
        def __init__(self, fx: _Optional[float] = ..., fy: _Optional[float] = ..., cx: _Optional[float] = ..., cy: _Optional[float] = ..., k0: _Optional[float] = ..., k1: _Optional[float] = ..., k2: _Optional[float] = ..., w: _Optional[int] = ..., h: _Optional[int] = ...) -> None: ...
    class CancelTrackingWithOcclusionOptions(_message.Message):
        __slots__ = ["activated", "min_inlier_ratio", "min_motion_continuity"]
        ACTIVATED_FIELD_NUMBER: _ClassVar[int]
        MIN_INLIER_RATIO_FIELD_NUMBER: _ClassVar[int]
        MIN_MOTION_CONTINUITY_FIELD_NUMBER: _ClassVar[int]
        activated: bool
        min_inlier_ratio: float
        min_motion_continuity: float
        def __init__(self, activated: bool = ..., min_motion_continuity: _Optional[float] = ..., min_inlier_ratio: _Optional[float] = ...) -> None: ...
    class IrlsInitialization(_message.Message):
        __slots__ = ["activated", "cutoff", "rounds"]
        ACTIVATED_FIELD_NUMBER: _ClassVar[int]
        CUTOFF_FIELD_NUMBER: _ClassVar[int]
        ROUNDS_FIELD_NUMBER: _ClassVar[int]
        activated: bool
        cutoff: float
        rounds: int
        def __init__(self, activated: bool = ..., rounds: _Optional[int] = ..., cutoff: _Optional[float] = ...) -> None: ...
    BACKGROUND_DISCRIMINATION_HIGH_LEVEL_FIELD_NUMBER: _ClassVar[int]
    BACKGROUND_DISCRIMINATION_LOW_LEVEL_FIELD_NUMBER: _ClassVar[int]
    BOX_SIMILARITY_MAX_ROTATION_FIELD_NUMBER: _ClassVar[int]
    BOX_SIMILARITY_MAX_SCALE_FIELD_NUMBER: _ClassVar[int]
    CAMERA_INTRINSICS_FIELD_NUMBER: _ClassVar[int]
    CANCEL_TRACKING_WITH_OCCLUSION_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    COMPUTE_SPATIAL_PRIOR_FIELD_NUMBER: _ClassVar[int]
    DISPARITY_DECAY_FIELD_NUMBER: _ClassVar[int]
    EXPANSION_SIZE_FIELD_NUMBER: _ClassVar[int]
    FORCED_PNP_TRACKING_FIELD_NUMBER: _ClassVar[int]
    HIGH_KINETIC_ENERGY_FIELD_NUMBER: _ClassVar[int]
    INLIER_CENTER_RELATIVE_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    INLIER_HIGH_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    INLIER_LOW_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    INLIER_SPRING_FORCE_FIELD_NUMBER: _ClassVar[int]
    IRLS_INITIALIZATION_FIELD_NUMBER: _ClassVar[int]
    IRLS_ITERATIONS_FIELD_NUMBER: _ClassVar[int]
    KINETIC_CENTER_RELATIVE_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    KINETIC_ENERGY_DECAY_FIELD_NUMBER: _ClassVar[int]
    KINETIC_SPRING_FORCE_FIELD_NUMBER: _ClassVar[int]
    KINETIC_SPRING_FORCE_MIN_KINETIC_ENERGY_FIELD_NUMBER: _ClassVar[int]
    LOW_KINETIC_ENERGY_FIELD_NUMBER: _ClassVar[int]
    MAX_TRACK_FAILURES_FIELD_NUMBER: _ClassVar[int]
    MIN_MOTION_SIGMA_FIELD_NUMBER: _ClassVar[int]
    MOTION_DISPARITY_HIGH_LEVEL_FIELD_NUMBER: _ClassVar[int]
    MOTION_DISPARITY_LOW_LEVEL_FIELD_NUMBER: _ClassVar[int]
    MOTION_PRIOR_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    OBJECT_SIMILARITY_MIN_CONTD_INLIERS_FIELD_NUMBER: _ClassVar[int]
    PRIOR_WEIGHT_INCREASE_FIELD_NUMBER: _ClassVar[int]
    QUAD_HOMOGRAPHY_MAX_ROTATION_FIELD_NUMBER: _ClassVar[int]
    QUAD_HOMOGRAPHY_MAX_SCALE_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_MOTION_SIGMA_FIELD_NUMBER: _ClassVar[int]
    RETURN_INTERNAL_STATE_FIELD_NUMBER: _ClassVar[int]
    SPATIAL_SIGMA_FIELD_NUMBER: _ClassVar[int]
    STATIC_MOTION_TEMPORAL_RATIO_FIELD_NUMBER: _ClassVar[int]
    TRACKING_DEGREES_FIELD_NUMBER: _ClassVar[int]
    TRACKING_DEGREE_CAMERA_PERSPECTIVE: TrackStepOptions.TrackingDegrees
    TRACKING_DEGREE_CAMERA_ROTATION: TrackStepOptions.TrackingDegrees
    TRACKING_DEGREE_CAMERA_ROTATION_SCALE: TrackStepOptions.TrackingDegrees
    TRACKING_DEGREE_CAMERA_SCALE: TrackStepOptions.TrackingDegrees
    TRACKING_DEGREE_OBJECT_PERSPECTIVE: TrackStepOptions.TrackingDegrees
    TRACKING_DEGREE_OBJECT_ROTATION: TrackStepOptions.TrackingDegrees
    TRACKING_DEGREE_OBJECT_ROTATION_SCALE: TrackStepOptions.TrackingDegrees
    TRACKING_DEGREE_OBJECT_SCALE: TrackStepOptions.TrackingDegrees
    TRACKING_DEGREE_TRANSLATION: TrackStepOptions.TrackingDegrees
    TRACK_OBJECT_AND_CAMERA_FIELD_NUMBER: _ClassVar[int]
    USE_POST_ESTIMATION_WEIGHTS_FOR_STATE_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_UPDATE_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    background_discrimination_high_level: float
    background_discrimination_low_level: float
    box_similarity_max_rotation: float
    box_similarity_max_scale: float
    camera_intrinsics: TrackStepOptions.CameraIntrinsics
    cancel_tracking_with_occlusion_options: TrackStepOptions.CancelTrackingWithOcclusionOptions
    compute_spatial_prior: bool
    disparity_decay: float
    expansion_size: float
    forced_pnp_tracking: bool
    high_kinetic_energy: float
    inlier_center_relative_distance: float
    inlier_high_weight: float
    inlier_low_weight: float
    inlier_spring_force: float
    irls_initialization: TrackStepOptions.IrlsInitialization
    irls_iterations: int
    kinetic_center_relative_distance: float
    kinetic_energy_decay: float
    kinetic_spring_force: float
    kinetic_spring_force_min_kinetic_energy: float
    low_kinetic_energy: float
    max_track_failures: int
    min_motion_sigma: float
    motion_disparity_high_level: float
    motion_disparity_low_level: float
    motion_prior_weight: float
    object_similarity_min_contd_inliers: int
    prior_weight_increase: float
    quad_homography_max_rotation: float
    quad_homography_max_scale: float
    relative_motion_sigma: float
    return_internal_state: bool
    spatial_sigma: float
    static_motion_temporal_ratio: float
    track_object_and_camera: bool
    tracking_degrees: TrackStepOptions.TrackingDegrees
    use_post_estimation_weights_for_state: bool
    velocity_update_weight: float
    def __init__(self, tracking_degrees: _Optional[_Union[TrackStepOptions.TrackingDegrees, str]] = ..., track_object_and_camera: bool = ..., irls_iterations: _Optional[int] = ..., spatial_sigma: _Optional[float] = ..., min_motion_sigma: _Optional[float] = ..., relative_motion_sigma: _Optional[float] = ..., motion_disparity_low_level: _Optional[float] = ..., motion_disparity_high_level: _Optional[float] = ..., disparity_decay: _Optional[float] = ..., motion_prior_weight: _Optional[float] = ..., background_discrimination_low_level: _Optional[float] = ..., background_discrimination_high_level: _Optional[float] = ..., inlier_center_relative_distance: _Optional[float] = ..., inlier_spring_force: _Optional[float] = ..., kinetic_center_relative_distance: _Optional[float] = ..., kinetic_spring_force: _Optional[float] = ..., kinetic_spring_force_min_kinetic_energy: _Optional[float] = ..., velocity_update_weight: _Optional[float] = ..., max_track_failures: _Optional[int] = ..., expansion_size: _Optional[float] = ..., inlier_low_weight: _Optional[float] = ..., inlier_high_weight: _Optional[float] = ..., kinetic_energy_decay: _Optional[float] = ..., prior_weight_increase: _Optional[float] = ..., low_kinetic_energy: _Optional[float] = ..., high_kinetic_energy: _Optional[float] = ..., return_internal_state: bool = ..., use_post_estimation_weights_for_state: bool = ..., compute_spatial_prior: bool = ..., irls_initialization: _Optional[_Union[TrackStepOptions.IrlsInitialization, _Mapping]] = ..., static_motion_temporal_ratio: _Optional[float] = ..., cancel_tracking_with_occlusion_options: _Optional[_Union[TrackStepOptions.CancelTrackingWithOcclusionOptions, _Mapping]] = ..., object_similarity_min_contd_inliers: _Optional[int] = ..., box_similarity_max_scale: _Optional[float] = ..., box_similarity_max_rotation: _Optional[float] = ..., quad_homography_max_scale: _Optional[float] = ..., quad_homography_max_rotation: _Optional[float] = ..., camera_intrinsics: _Optional[_Union[TrackStepOptions.CameraIntrinsics, _Mapping]] = ..., forced_pnp_tracking: bool = ...) -> None: ...
