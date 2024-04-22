from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BinaryFeatureDescriptor(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class PatchDescriptor(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, data: _Optional[_Iterable[float]] = ...) -> None: ...

class RegionFlowFeature(_message.Message):
    __slots__ = ["binary_feature_descriptor", "corner_response", "dx", "dy", "feature_descriptor", "feature_id", "feature_match_descriptor", "flags", "internal_irls", "irls_weight", "label", "octave", "track_id", "tracking_error", "x", "y"]
    class Flags(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BINARY_FEATURE_DESCRIPTOR_FIELD_NUMBER: _ClassVar[int]
    CORNER_RESPONSE_FIELD_NUMBER: _ClassVar[int]
    DX_FIELD_NUMBER: _ClassVar[int]
    DY_FIELD_NUMBER: _ClassVar[int]
    Extensions: _python_message._ExtensionDict
    FEATURE_DESCRIPTOR_FIELD_NUMBER: _ClassVar[int]
    FEATURE_ID_FIELD_NUMBER: _ClassVar[int]
    FEATURE_MATCH_DESCRIPTOR_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    INTERNAL_IRLS_FIELD_NUMBER: _ClassVar[int]
    IRLS_WEIGHT_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    OCTAVE_FIELD_NUMBER: _ClassVar[int]
    REGION_FLOW_FLAG_BROKEN_TRACK: RegionFlowFeature.Flags
    REGION_FLOW_FLAG_UNKNOWN: RegionFlowFeature.Flags
    TRACKING_ERROR_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    binary_feature_descriptor: BinaryFeatureDescriptor
    corner_response: float
    dx: float
    dy: float
    feature_descriptor: PatchDescriptor
    feature_id: int
    feature_match_descriptor: PatchDescriptor
    flags: int
    internal_irls: TemporalIRLSSmoothing
    irls_weight: float
    label: str
    octave: int
    track_id: int
    tracking_error: float
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., dx: _Optional[float] = ..., dy: _Optional[float] = ..., track_id: _Optional[int] = ..., tracking_error: _Optional[float] = ..., irls_weight: _Optional[float] = ..., corner_response: _Optional[float] = ..., feature_descriptor: _Optional[_Union[PatchDescriptor, _Mapping]] = ..., feature_match_descriptor: _Optional[_Union[PatchDescriptor, _Mapping]] = ..., internal_irls: _Optional[_Union[TemporalIRLSSmoothing, _Mapping]] = ..., label: _Optional[str] = ..., flags: _Optional[int] = ..., feature_id: _Optional[int] = ..., octave: _Optional[int] = ..., binary_feature_descriptor: _Optional[_Union[BinaryFeatureDescriptor, _Mapping]] = ...) -> None: ...

class RegionFlowFeatureList(_message.Message):
    __slots__ = ["actively_discarded_tracked_ids", "blur_score", "distance_from_border", "feature", "frac_long_features_rejected", "frame_height", "frame_width", "is_duplicated", "long_tracks", "match_frame", "timestamp_usec", "unstable", "visual_consistency"]
    ACTIVELY_DISCARDED_TRACKED_IDS_FIELD_NUMBER: _ClassVar[int]
    BLUR_SCORE_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FROM_BORDER_FIELD_NUMBER: _ClassVar[int]
    FEATURE_FIELD_NUMBER: _ClassVar[int]
    FRAC_LONG_FEATURES_REJECTED_FIELD_NUMBER: _ClassVar[int]
    FRAME_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FRAME_WIDTH_FIELD_NUMBER: _ClassVar[int]
    IS_DUPLICATED_FIELD_NUMBER: _ClassVar[int]
    LONG_TRACKS_FIELD_NUMBER: _ClassVar[int]
    MATCH_FRAME_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_USEC_FIELD_NUMBER: _ClassVar[int]
    UNSTABLE_FIELD_NUMBER: _ClassVar[int]
    VISUAL_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    actively_discarded_tracked_ids: _containers.RepeatedScalarFieldContainer[int]
    blur_score: float
    distance_from_border: int
    feature: _containers.RepeatedCompositeFieldContainer[RegionFlowFeature]
    frac_long_features_rejected: float
    frame_height: int
    frame_width: int
    is_duplicated: bool
    long_tracks: bool
    match_frame: int
    timestamp_usec: int
    unstable: bool
    visual_consistency: float
    def __init__(self, feature: _Optional[_Iterable[_Union[RegionFlowFeature, _Mapping]]] = ..., frame_width: _Optional[int] = ..., frame_height: _Optional[int] = ..., unstable: bool = ..., distance_from_border: _Optional[int] = ..., blur_score: _Optional[float] = ..., long_tracks: bool = ..., frac_long_features_rejected: _Optional[float] = ..., visual_consistency: _Optional[float] = ..., timestamp_usec: _Optional[int] = ..., match_frame: _Optional[int] = ..., is_duplicated: bool = ..., actively_discarded_tracked_ids: _Optional[_Iterable[int]] = ...) -> None: ...

class RegionFlowFrame(_message.Message):
    __slots__ = ["block_descriptor", "blur_score", "frame_height", "frame_width", "num_total_features", "region_flow", "unstable_frame"]
    class BlockDescriptor(_message.Message):
        __slots__ = ["block_height", "block_width", "num_blocks_x", "num_blocks_y"]
        BLOCK_HEIGHT_FIELD_NUMBER: _ClassVar[int]
        BLOCK_WIDTH_FIELD_NUMBER: _ClassVar[int]
        NUM_BLOCKS_X_FIELD_NUMBER: _ClassVar[int]
        NUM_BLOCKS_Y_FIELD_NUMBER: _ClassVar[int]
        block_height: int
        block_width: int
        num_blocks_x: int
        num_blocks_y: int
        def __init__(self, block_width: _Optional[int] = ..., block_height: _Optional[int] = ..., num_blocks_x: _Optional[int] = ..., num_blocks_y: _Optional[int] = ...) -> None: ...
    class RegionFlow(_message.Message):
        __slots__ = ["centroid_x", "centroid_y", "feature", "flow_x", "flow_y", "region_id"]
        CENTROID_X_FIELD_NUMBER: _ClassVar[int]
        CENTROID_Y_FIELD_NUMBER: _ClassVar[int]
        Extensions: _python_message._ExtensionDict
        FEATURE_FIELD_NUMBER: _ClassVar[int]
        FLOW_X_FIELD_NUMBER: _ClassVar[int]
        FLOW_Y_FIELD_NUMBER: _ClassVar[int]
        REGION_ID_FIELD_NUMBER: _ClassVar[int]
        centroid_x: float
        centroid_y: float
        feature: _containers.RepeatedCompositeFieldContainer[RegionFlowFeature]
        flow_x: float
        flow_y: float
        region_id: int
        def __init__(self, region_id: _Optional[int] = ..., centroid_x: _Optional[float] = ..., centroid_y: _Optional[float] = ..., flow_x: _Optional[float] = ..., flow_y: _Optional[float] = ..., feature: _Optional[_Iterable[_Union[RegionFlowFeature, _Mapping]]] = ...) -> None: ...
    BLOCK_DESCRIPTOR_FIELD_NUMBER: _ClassVar[int]
    BLUR_SCORE_FIELD_NUMBER: _ClassVar[int]
    Extensions: _python_message._ExtensionDict
    FRAME_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FRAME_WIDTH_FIELD_NUMBER: _ClassVar[int]
    NUM_TOTAL_FEATURES_FIELD_NUMBER: _ClassVar[int]
    REGION_FLOW_FIELD_NUMBER: _ClassVar[int]
    UNSTABLE_FRAME_FIELD_NUMBER: _ClassVar[int]
    block_descriptor: RegionFlowFrame.BlockDescriptor
    blur_score: float
    frame_height: int
    frame_width: int
    num_total_features: int
    region_flow: _containers.RepeatedCompositeFieldContainer[RegionFlowFrame.RegionFlow]
    unstable_frame: bool
    def __init__(self, region_flow: _Optional[_Iterable[_Union[RegionFlowFrame.RegionFlow, _Mapping]]] = ..., num_total_features: _Optional[int] = ..., unstable_frame: bool = ..., blur_score: _Optional[float] = ..., frame_width: _Optional[int] = ..., frame_height: _Optional[int] = ..., block_descriptor: _Optional[_Union[RegionFlowFrame.BlockDescriptor, _Mapping]] = ...) -> None: ...

class SalientPoint(_message.Message):
    __slots__ = ["angle", "bottom", "left", "norm_major", "norm_minor", "norm_point_x", "norm_point_y", "right", "top", "type", "weight"]
    class SalientPointType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ANGLE_FIELD_NUMBER: _ClassVar[int]
    BOTTOM_FIELD_NUMBER: _ClassVar[int]
    Extensions: _python_message._ExtensionDict
    LEFT_FIELD_NUMBER: _ClassVar[int]
    NORM_MAJOR_FIELD_NUMBER: _ClassVar[int]
    NORM_MINOR_FIELD_NUMBER: _ClassVar[int]
    NORM_POINT_X_FIELD_NUMBER: _ClassVar[int]
    NORM_POINT_Y_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    SALIENT_POINT_TYPE_EXCLUDE_LEFT: SalientPoint.SalientPointType
    SALIENT_POINT_TYPE_EXCLUDE_RIGHT: SalientPoint.SalientPointType
    SALIENT_POINT_TYPE_INCLUDE: SalientPoint.SalientPointType
    SALIENT_POINT_TYPE_UNKNOWN: SalientPoint.SalientPointType
    TOP_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    angle: float
    bottom: float
    left: float
    norm_major: float
    norm_minor: float
    norm_point_x: float
    norm_point_y: float
    right: float
    top: float
    type: SalientPoint.SalientPointType
    weight: float
    def __init__(self, norm_point_x: _Optional[float] = ..., norm_point_y: _Optional[float] = ..., type: _Optional[_Union[SalientPoint.SalientPointType, str]] = ..., left: _Optional[float] = ..., bottom: _Optional[float] = ..., right: _Optional[float] = ..., top: _Optional[float] = ..., weight: _Optional[float] = ..., norm_major: _Optional[float] = ..., norm_minor: _Optional[float] = ..., angle: _Optional[float] = ...) -> None: ...

class SalientPointFrame(_message.Message):
    __slots__ = ["point"]
    Extensions: _python_message._ExtensionDict
    POINT_FIELD_NUMBER: _ClassVar[int]
    point: _containers.RepeatedCompositeFieldContainer[SalientPoint]
    def __init__(self, point: _Optional[_Iterable[_Union[SalientPoint, _Mapping]]] = ...) -> None: ...

class TemporalIRLSSmoothing(_message.Message):
    __slots__ = ["value_sum", "weight_sum"]
    VALUE_SUM_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_SUM_FIELD_NUMBER: _ClassVar[int]
    value_sum: float
    weight_sum: float
    def __init__(self, weight_sum: _Optional[float] = ..., value_sum: _Optional[float] = ...) -> None: ...
