from mediapipe.util.tracking import box_tracker_pb2 as _box_tracker_pb2
from mediapipe.util.tracking import region_flow_pb2 as _region_flow_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BoxDetectorIndex(_message.Message):
    __slots__ = ["box_entry"]
    class BoxEntry(_message.Message):
        __slots__ = ["frame_entry"]
        class FrameEntry(_message.Message):
            __slots__ = ["box", "descriptors", "keypoints"]
            BOX_FIELD_NUMBER: _ClassVar[int]
            DESCRIPTORS_FIELD_NUMBER: _ClassVar[int]
            KEYPOINTS_FIELD_NUMBER: _ClassVar[int]
            box: _box_tracker_pb2.TimedBoxProto
            descriptors: _containers.RepeatedCompositeFieldContainer[_region_flow_pb2.BinaryFeatureDescriptor]
            keypoints: _containers.RepeatedScalarFieldContainer[float]
            def __init__(self, box: _Optional[_Union[_box_tracker_pb2.TimedBoxProto, _Mapping]] = ..., keypoints: _Optional[_Iterable[float]] = ..., descriptors: _Optional[_Iterable[_Union[_region_flow_pb2.BinaryFeatureDescriptor, _Mapping]]] = ...) -> None: ...
        FRAME_ENTRY_FIELD_NUMBER: _ClassVar[int]
        frame_entry: _containers.RepeatedCompositeFieldContainer[BoxDetectorIndex.BoxEntry.FrameEntry]
        def __init__(self, frame_entry: _Optional[_Iterable[_Union[BoxDetectorIndex.BoxEntry.FrameEntry, _Mapping]]] = ...) -> None: ...
    BOX_ENTRY_FIELD_NUMBER: _ClassVar[int]
    box_entry: _containers.RepeatedCompositeFieldContainer[BoxDetectorIndex.BoxEntry]
    def __init__(self, box_entry: _Optional[_Iterable[_Union[BoxDetectorIndex.BoxEntry, _Mapping]]] = ...) -> None: ...

class BoxDetectorOptions(_message.Message):
    __slots__ = ["descriptor_dims", "detect_every_n_frame", "detect_out_of_fov", "image_query_settings", "index_type", "max_match_distance", "max_perspective_factor", "min_num_correspondence", "ransac_reprojection_threshold"]
    class IndexType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class ImageQuerySettings(_message.Message):
        __slots__ = ["max_features", "max_pyramid_levels", "pyramid_bottom_size", "pyramid_scale_factor"]
        MAX_FEATURES_FIELD_NUMBER: _ClassVar[int]
        MAX_PYRAMID_LEVELS_FIELD_NUMBER: _ClassVar[int]
        PYRAMID_BOTTOM_SIZE_FIELD_NUMBER: _ClassVar[int]
        PYRAMID_SCALE_FACTOR_FIELD_NUMBER: _ClassVar[int]
        max_features: int
        max_pyramid_levels: int
        pyramid_bottom_size: int
        pyramid_scale_factor: float
        def __init__(self, pyramid_bottom_size: _Optional[int] = ..., pyramid_scale_factor: _Optional[float] = ..., max_pyramid_levels: _Optional[int] = ..., max_features: _Optional[int] = ...) -> None: ...
    DESCRIPTOR_DIMS_FIELD_NUMBER: _ClassVar[int]
    DETECT_EVERY_N_FRAME_FIELD_NUMBER: _ClassVar[int]
    DETECT_OUT_OF_FOV_FIELD_NUMBER: _ClassVar[int]
    IMAGE_QUERY_SETTINGS_FIELD_NUMBER: _ClassVar[int]
    INDEX_TYPE_FIELD_NUMBER: _ClassVar[int]
    INDEX_UNSPECIFIED: BoxDetectorOptions.IndexType
    MAX_MATCH_DISTANCE_FIELD_NUMBER: _ClassVar[int]
    MAX_PERSPECTIVE_FACTOR_FIELD_NUMBER: _ClassVar[int]
    MIN_NUM_CORRESPONDENCE_FIELD_NUMBER: _ClassVar[int]
    OPENCV_BF: BoxDetectorOptions.IndexType
    RANSAC_REPROJECTION_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    descriptor_dims: int
    detect_every_n_frame: int
    detect_out_of_fov: bool
    image_query_settings: BoxDetectorOptions.ImageQuerySettings
    index_type: BoxDetectorOptions.IndexType
    max_match_distance: float
    max_perspective_factor: float
    min_num_correspondence: int
    ransac_reprojection_threshold: float
    def __init__(self, index_type: _Optional[_Union[BoxDetectorOptions.IndexType, str]] = ..., detect_every_n_frame: _Optional[int] = ..., detect_out_of_fov: bool = ..., image_query_settings: _Optional[_Union[BoxDetectorOptions.ImageQuerySettings, _Mapping]] = ..., descriptor_dims: _Optional[int] = ..., min_num_correspondence: _Optional[int] = ..., ransac_reprojection_threshold: _Optional[float] = ..., max_match_distance: _Optional[float] = ..., max_perspective_factor: _Optional[float] = ...) -> None: ...
