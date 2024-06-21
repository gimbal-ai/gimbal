from google.protobuf import wrappers_pb2 as _wrappers_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ByteTrackCalculatorOptions(_message.Message):
    __slots__ = ["high_thresh", "match_thresh", "max_frames_lost", "track_thresh"]
    HIGH_THRESH_FIELD_NUMBER: _ClassVar[int]
    MATCH_THRESH_FIELD_NUMBER: _ClassVar[int]
    MAX_FRAMES_LOST_FIELD_NUMBER: _ClassVar[int]
    TRACK_THRESH_FIELD_NUMBER: _ClassVar[int]
    high_thresh: _wrappers_pb2.FloatValue
    match_thresh: _wrappers_pb2.FloatValue
    max_frames_lost: _wrappers_pb2.Int32Value
    track_thresh: _wrappers_pb2.FloatValue
    def __init__(self, max_frames_lost: _Optional[_Union[_wrappers_pb2.Int32Value, _Mapping]] = ..., track_thresh: _Optional[_Union[_wrappers_pb2.FloatValue, _Mapping]] = ..., high_thresh: _Optional[_Union[_wrappers_pb2.FloatValue, _Mapping]] = ..., match_thresh: _Optional[_Union[_wrappers_pb2.FloatValue, _Mapping]] = ...) -> None: ...
