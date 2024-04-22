from mediapipe.util.tracking import tracking_pb2 as _tracking_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BoxTrackerOptions(_message.Message):
    __slots__ = ["cache_file_format", "caching_chunk_size_msec", "num_tracking_workers", "read_chunk_timeout_msec", "record_path_states", "track_step_options"]
    CACHE_FILE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    CACHING_CHUNK_SIZE_MSEC_FIELD_NUMBER: _ClassVar[int]
    NUM_TRACKING_WORKERS_FIELD_NUMBER: _ClassVar[int]
    READ_CHUNK_TIMEOUT_MSEC_FIELD_NUMBER: _ClassVar[int]
    RECORD_PATH_STATES_FIELD_NUMBER: _ClassVar[int]
    TRACK_STEP_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    cache_file_format: str
    caching_chunk_size_msec: int
    num_tracking_workers: int
    read_chunk_timeout_msec: int
    record_path_states: bool
    track_step_options: _tracking_pb2.TrackStepOptions
    def __init__(self, caching_chunk_size_msec: _Optional[int] = ..., cache_file_format: _Optional[str] = ..., num_tracking_workers: _Optional[int] = ..., read_chunk_timeout_msec: _Optional[int] = ..., record_path_states: bool = ..., track_step_options: _Optional[_Union[_tracking_pb2.TrackStepOptions, _Mapping]] = ...) -> None: ...

class TimedBoxProto(_message.Message):
    __slots__ = ["aspect_ratio", "bottom", "confidence", "id", "label", "left", "quad", "reacquisition", "request_grouping", "right", "rotation", "time_msec", "top"]
    ASPECT_RATIO_FIELD_NUMBER: _ClassVar[int]
    BOTTOM_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    LEFT_FIELD_NUMBER: _ClassVar[int]
    QUAD_FIELD_NUMBER: _ClassVar[int]
    REACQUISITION_FIELD_NUMBER: _ClassVar[int]
    REQUEST_GROUPING_FIELD_NUMBER: _ClassVar[int]
    RIGHT_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    TIME_MSEC_FIELD_NUMBER: _ClassVar[int]
    TOP_FIELD_NUMBER: _ClassVar[int]
    aspect_ratio: float
    bottom: float
    confidence: float
    id: int
    label: str
    left: float
    quad: _tracking_pb2.MotionBoxState.Quad
    reacquisition: bool
    request_grouping: bool
    right: float
    rotation: float
    time_msec: int
    top: float
    def __init__(self, top: _Optional[float] = ..., left: _Optional[float] = ..., bottom: _Optional[float] = ..., right: _Optional[float] = ..., rotation: _Optional[float] = ..., quad: _Optional[_Union[_tracking_pb2.MotionBoxState.Quad, _Mapping]] = ..., time_msec: _Optional[int] = ..., id: _Optional[int] = ..., label: _Optional[str] = ..., confidence: _Optional[float] = ..., aspect_ratio: _Optional[float] = ..., reacquisition: bool = ..., request_grouping: bool = ...) -> None: ...

class TimedBoxProtoList(_message.Message):
    __slots__ = ["box"]
    BOX_FIELD_NUMBER: _ClassVar[int]
    box: _containers.RepeatedCompositeFieldContainer[TimedBoxProto]
    def __init__(self, box: _Optional[_Iterable[_Union[TimedBoxProto, _Mapping]]] = ...) -> None: ...
