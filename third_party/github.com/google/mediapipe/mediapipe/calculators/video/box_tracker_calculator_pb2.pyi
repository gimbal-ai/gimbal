from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util.tracking import box_tracker_pb2 as _box_tracker_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BoxTrackerCalculatorOptions(_message.Message):
    __slots__ = ["initial_position", "start_pos_transition_frames", "streaming_track_data_cache_size", "tracker_options", "visualize_internal_state", "visualize_state", "visualize_tracking_data"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    INITIAL_POSITION_FIELD_NUMBER: _ClassVar[int]
    START_POS_TRANSITION_FRAMES_FIELD_NUMBER: _ClassVar[int]
    STREAMING_TRACK_DATA_CACHE_SIZE_FIELD_NUMBER: _ClassVar[int]
    TRACKER_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    VISUALIZE_INTERNAL_STATE_FIELD_NUMBER: _ClassVar[int]
    VISUALIZE_STATE_FIELD_NUMBER: _ClassVar[int]
    VISUALIZE_TRACKING_DATA_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    initial_position: _box_tracker_pb2.TimedBoxProtoList
    start_pos_transition_frames: int
    streaming_track_data_cache_size: int
    tracker_options: _box_tracker_pb2.BoxTrackerOptions
    visualize_internal_state: bool
    visualize_state: bool
    visualize_tracking_data: bool
    def __init__(self, tracker_options: _Optional[_Union[_box_tracker_pb2.BoxTrackerOptions, _Mapping]] = ..., initial_position: _Optional[_Union[_box_tracker_pb2.TimedBoxProtoList, _Mapping]] = ..., visualize_tracking_data: bool = ..., visualize_state: bool = ..., visualize_internal_state: bool = ..., streaming_track_data_cache_size: _Optional[int] = ..., start_pos_transition_frames: _Optional[int] = ...) -> None: ...
