from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AudioDecoderOptions(_message.Message):
    __slots__ = ["audio_stream", "end_time", "start_time"]
    AUDIO_STREAM_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    audio_stream: _containers.RepeatedCompositeFieldContainer[AudioStreamOptions]
    end_time: float
    ext: _descriptor.FieldDescriptor
    start_time: float
    def __init__(self, audio_stream: _Optional[_Iterable[_Union[AudioStreamOptions, _Mapping]]] = ..., start_time: _Optional[float] = ..., end_time: _Optional[float] = ...) -> None: ...

class AudioStreamOptions(_message.Message):
    __slots__ = ["allow_missing", "correct_pts_for_rollover", "ignore_decode_failures", "output_regressing_timestamps", "stream_index"]
    ALLOW_MISSING_FIELD_NUMBER: _ClassVar[int]
    CORRECT_PTS_FOR_ROLLOVER_FIELD_NUMBER: _ClassVar[int]
    IGNORE_DECODE_FAILURES_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_REGRESSING_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    STREAM_INDEX_FIELD_NUMBER: _ClassVar[int]
    allow_missing: bool
    correct_pts_for_rollover: bool
    ignore_decode_failures: bool
    output_regressing_timestamps: bool
    stream_index: int
    def __init__(self, stream_index: _Optional[int] = ..., allow_missing: bool = ..., ignore_decode_failures: bool = ..., output_regressing_timestamps: bool = ..., correct_pts_for_rollover: bool = ...) -> None: ...
