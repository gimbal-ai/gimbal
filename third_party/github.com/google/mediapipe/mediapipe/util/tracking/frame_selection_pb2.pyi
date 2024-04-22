from mediapipe.util.tracking import camera_motion_pb2 as _camera_motion_pb2
from mediapipe.util.tracking import frame_selection_solution_evaluator_pb2 as _frame_selection_solution_evaluator_pb2
from mediapipe.util.tracking import region_flow_pb2 as _region_flow_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FrameSelectionCriterion(_message.Message):
    __slots__ = ["bandwidth_frames", "max_output_frames", "sampling_rate", "search_radius_frames", "solution_evaluator"]
    BANDWIDTH_FRAMES_FIELD_NUMBER: _ClassVar[int]
    MAX_OUTPUT_FRAMES_FIELD_NUMBER: _ClassVar[int]
    SAMPLING_RATE_FIELD_NUMBER: _ClassVar[int]
    SEARCH_RADIUS_FRAMES_FIELD_NUMBER: _ClassVar[int]
    SOLUTION_EVALUATOR_FIELD_NUMBER: _ClassVar[int]
    bandwidth_frames: float
    max_output_frames: int
    sampling_rate: int
    search_radius_frames: int
    solution_evaluator: _frame_selection_solution_evaluator_pb2.FrameSelectionSolutionEvaluatorType
    def __init__(self, sampling_rate: _Optional[int] = ..., bandwidth_frames: _Optional[float] = ..., search_radius_frames: _Optional[int] = ..., solution_evaluator: _Optional[_Union[_frame_selection_solution_evaluator_pb2.FrameSelectionSolutionEvaluatorType, _Mapping]] = ..., max_output_frames: _Optional[int] = ...) -> None: ...

class FrameSelectionOptions(_message.Message):
    __slots__ = ["chunk_size", "criterion"]
    CHUNK_SIZE_FIELD_NUMBER: _ClassVar[int]
    CRITERION_FIELD_NUMBER: _ClassVar[int]
    chunk_size: int
    criterion: _containers.RepeatedCompositeFieldContainer[FrameSelectionCriterion]
    def __init__(self, criterion: _Optional[_Iterable[_Union[FrameSelectionCriterion, _Mapping]]] = ..., chunk_size: _Optional[int] = ...) -> None: ...

class FrameSelectionResult(_message.Message):
    __slots__ = ["camera_motion", "features", "frame_idx", "processed_from_timestamp", "timestamp"]
    CAMERA_MOTION_FIELD_NUMBER: _ClassVar[int]
    FEATURES_FIELD_NUMBER: _ClassVar[int]
    FRAME_IDX_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_FROM_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    camera_motion: _camera_motion_pb2.CameraMotion
    features: _region_flow_pb2.RegionFlowFeatureList
    frame_idx: int
    processed_from_timestamp: int
    timestamp: int
    def __init__(self, timestamp: _Optional[int] = ..., frame_idx: _Optional[int] = ..., camera_motion: _Optional[_Union[_camera_motion_pb2.CameraMotion, _Mapping]] = ..., features: _Optional[_Union[_region_flow_pb2.RegionFlowFeatureList, _Mapping]] = ..., processed_from_timestamp: _Optional[int] = ...) -> None: ...

class FrameSelectionTimestamp(_message.Message):
    __slots__ = ["frame_idx", "processed_from_timestamp", "timestamp"]
    FRAME_IDX_FIELD_NUMBER: _ClassVar[int]
    PROCESSED_FROM_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    frame_idx: int
    processed_from_timestamp: int
    timestamp: int
    def __init__(self, timestamp: _Optional[int] = ..., frame_idx: _Optional[int] = ..., processed_from_timestamp: _Optional[int] = ...) -> None: ...
