from mediapipe.util.tracking import motion_models_pb2 as _motion_models_pb2
from mediapipe.util.tracking import region_flow_pb2 as _region_flow_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BinaryTrackingData(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class FlowPackagerOptions(_message.Message):
    __slots__ = ["binary_tracking_data_support", "domain_height", "domain_width", "high_fidelity_16bit_encode", "high_profile_reuse_threshold", "use_high_profile"]
    class HighProfileEncoding(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ADVANCE_FLAG: FlowPackagerOptions.HighProfileEncoding
    BINARY_TRACKING_DATA_SUPPORT_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_WIDTH_FIELD_NUMBER: _ClassVar[int]
    DOUBLE_INDEX_ENCODE: FlowPackagerOptions.HighProfileEncoding
    HIGH_FIDELITY_16BIT_ENCODE_FIELD_NUMBER: _ClassVar[int]
    HIGH_PROFILE_REUSE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    INDEX_MASK: FlowPackagerOptions.HighProfileEncoding
    USE_HIGH_PROFILE_FIELD_NUMBER: _ClassVar[int]
    binary_tracking_data_support: bool
    domain_height: int
    domain_width: int
    high_fidelity_16bit_encode: bool
    high_profile_reuse_threshold: float
    use_high_profile: bool
    def __init__(self, domain_width: _Optional[int] = ..., domain_height: _Optional[int] = ..., binary_tracking_data_support: bool = ..., use_high_profile: bool = ..., high_fidelity_16bit_encode: bool = ..., high_profile_reuse_threshold: _Optional[float] = ...) -> None: ...

class MetaData(_message.Message):
    __slots__ = ["num_frames", "track_offsets"]
    class TrackOffset(_message.Message):
        __slots__ = ["msec", "stream_offset"]
        MSEC_FIELD_NUMBER: _ClassVar[int]
        STREAM_OFFSET_FIELD_NUMBER: _ClassVar[int]
        msec: int
        stream_offset: int
        def __init__(self, msec: _Optional[int] = ..., stream_offset: _Optional[int] = ...) -> None: ...
    NUM_FRAMES_FIELD_NUMBER: _ClassVar[int]
    TRACK_OFFSETS_FIELD_NUMBER: _ClassVar[int]
    num_frames: int
    track_offsets: _containers.RepeatedCompositeFieldContainer[MetaData.TrackOffset]
    def __init__(self, num_frames: _Optional[int] = ..., track_offsets: _Optional[_Iterable[_Union[MetaData.TrackOffset, _Mapping]]] = ...) -> None: ...

class TrackingContainer(_message.Message):
    __slots__ = ["data", "header", "size", "version"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    HEADER_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    header: str
    size: int
    version: int
    def __init__(self, header: _Optional[str] = ..., version: _Optional[int] = ..., size: _Optional[int] = ..., data: _Optional[bytes] = ...) -> None: ...

class TrackingContainerFormat(_message.Message):
    __slots__ = ["meta_data", "term_data", "track_data"]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    TERM_DATA_FIELD_NUMBER: _ClassVar[int]
    TRACK_DATA_FIELD_NUMBER: _ClassVar[int]
    meta_data: TrackingContainer
    term_data: TrackingContainer
    track_data: _containers.RepeatedCompositeFieldContainer[TrackingContainer]
    def __init__(self, meta_data: _Optional[_Union[TrackingContainer, _Mapping]] = ..., track_data: _Optional[_Iterable[_Union[TrackingContainer, _Mapping]]] = ..., term_data: _Optional[_Union[TrackingContainer, _Mapping]] = ...) -> None: ...

class TrackingContainerProto(_message.Message):
    __slots__ = ["meta_data", "track_data"]
    META_DATA_FIELD_NUMBER: _ClassVar[int]
    TRACK_DATA_FIELD_NUMBER: _ClassVar[int]
    meta_data: MetaData
    track_data: _containers.RepeatedCompositeFieldContainer[BinaryTrackingData]
    def __init__(self, meta_data: _Optional[_Union[MetaData, _Mapping]] = ..., track_data: _Optional[_Iterable[_Union[BinaryTrackingData, _Mapping]]] = ...) -> None: ...

class TrackingData(_message.Message):
    __slots__ = ["average_motion_magnitude", "background_model", "domain_height", "domain_width", "frame_aspect", "frame_flags", "global_feature_count", "motion_data"]
    class FrameFlags(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class MotionData(_message.Message):
        __slots__ = ["actively_discarded_tracked_ids", "col_starts", "feature_descriptors", "num_elements", "row_indices", "track_id", "vector_data"]
        ACTIVELY_DISCARDED_TRACKED_IDS_FIELD_NUMBER: _ClassVar[int]
        COL_STARTS_FIELD_NUMBER: _ClassVar[int]
        FEATURE_DESCRIPTORS_FIELD_NUMBER: _ClassVar[int]
        NUM_ELEMENTS_FIELD_NUMBER: _ClassVar[int]
        ROW_INDICES_FIELD_NUMBER: _ClassVar[int]
        TRACK_ID_FIELD_NUMBER: _ClassVar[int]
        VECTOR_DATA_FIELD_NUMBER: _ClassVar[int]
        actively_discarded_tracked_ids: _containers.RepeatedScalarFieldContainer[int]
        col_starts: _containers.RepeatedScalarFieldContainer[int]
        feature_descriptors: _containers.RepeatedCompositeFieldContainer[_region_flow_pb2.BinaryFeatureDescriptor]
        num_elements: int
        row_indices: _containers.RepeatedScalarFieldContainer[int]
        track_id: _containers.RepeatedScalarFieldContainer[int]
        vector_data: _containers.RepeatedScalarFieldContainer[float]
        def __init__(self, num_elements: _Optional[int] = ..., vector_data: _Optional[_Iterable[float]] = ..., track_id: _Optional[_Iterable[int]] = ..., row_indices: _Optional[_Iterable[int]] = ..., col_starts: _Optional[_Iterable[int]] = ..., feature_descriptors: _Optional[_Iterable[_Union[_region_flow_pb2.BinaryFeatureDescriptor, _Mapping]]] = ..., actively_discarded_tracked_ids: _Optional[_Iterable[int]] = ...) -> None: ...
    AVERAGE_MOTION_MAGNITUDE_FIELD_NUMBER: _ClassVar[int]
    BACKGROUND_MODEL_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_WIDTH_FIELD_NUMBER: _ClassVar[int]
    FRAME_ASPECT_FIELD_NUMBER: _ClassVar[int]
    FRAME_FLAGS_FIELD_NUMBER: _ClassVar[int]
    GLOBAL_FEATURE_COUNT_FIELD_NUMBER: _ClassVar[int]
    MOTION_DATA_FIELD_NUMBER: _ClassVar[int]
    TRACKING_FLAG_BACKGROUND_UNSTABLE: TrackingData.FrameFlags
    TRACKING_FLAG_CHUNK_BOUNDARY: TrackingData.FrameFlags
    TRACKING_FLAG_DUPLICATED: TrackingData.FrameFlags
    TRACKING_FLAG_HIGH_FIDELITY_VECTORS: TrackingData.FrameFlags
    TRACKING_FLAG_PROFILE_BASELINE: TrackingData.FrameFlags
    TRACKING_FLAG_PROFILE_HIGH: TrackingData.FrameFlags
    average_motion_magnitude: float
    background_model: _motion_models_pb2.Homography
    domain_height: int
    domain_width: int
    frame_aspect: float
    frame_flags: int
    global_feature_count: int
    motion_data: TrackingData.MotionData
    def __init__(self, frame_flags: _Optional[int] = ..., domain_width: _Optional[int] = ..., domain_height: _Optional[int] = ..., frame_aspect: _Optional[float] = ..., background_model: _Optional[_Union[_motion_models_pb2.Homography, _Mapping]] = ..., motion_data: _Optional[_Union[TrackingData.MotionData, _Mapping]] = ..., global_feature_count: _Optional[int] = ..., average_motion_magnitude: _Optional[float] = ...) -> None: ...

class TrackingDataChunk(_message.Message):
    __slots__ = ["first_chunk", "item", "last_chunk"]
    class Item(_message.Message):
        __slots__ = ["frame_idx", "prev_timestamp_usec", "timestamp_usec", "tracking_data"]
        FRAME_IDX_FIELD_NUMBER: _ClassVar[int]
        PREV_TIMESTAMP_USEC_FIELD_NUMBER: _ClassVar[int]
        TIMESTAMP_USEC_FIELD_NUMBER: _ClassVar[int]
        TRACKING_DATA_FIELD_NUMBER: _ClassVar[int]
        frame_idx: int
        prev_timestamp_usec: int
        timestamp_usec: int
        tracking_data: TrackingData
        def __init__(self, tracking_data: _Optional[_Union[TrackingData, _Mapping]] = ..., frame_idx: _Optional[int] = ..., timestamp_usec: _Optional[int] = ..., prev_timestamp_usec: _Optional[int] = ...) -> None: ...
    FIRST_CHUNK_FIELD_NUMBER: _ClassVar[int]
    ITEM_FIELD_NUMBER: _ClassVar[int]
    LAST_CHUNK_FIELD_NUMBER: _ClassVar[int]
    first_chunk: bool
    item: _containers.RepeatedCompositeFieldContainer[TrackingDataChunk.Item]
    last_chunk: bool
    def __init__(self, item: _Optional[_Iterable[_Union[TrackingDataChunk.Item, _Mapping]]] = ..., last_chunk: bool = ..., first_chunk: bool = ...) -> None: ...
