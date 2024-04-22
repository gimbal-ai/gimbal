from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MultiStreamTimeSeriesHeader(_message.Message):
    __slots__ = ["num_streams", "time_series_header"]
    NUM_STREAMS_FIELD_NUMBER: _ClassVar[int]
    TIME_SERIES_HEADER_FIELD_NUMBER: _ClassVar[int]
    num_streams: int
    time_series_header: TimeSeriesHeader
    def __init__(self, time_series_header: _Optional[_Union[TimeSeriesHeader, _Mapping]] = ..., num_streams: _Optional[int] = ...) -> None: ...

class TimeSeriesHeader(_message.Message):
    __slots__ = ["audio_sample_rate", "num_channels", "num_samples", "packet_rate", "sample_rate"]
    AUDIO_SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    Extensions: _python_message._ExtensionDict
    NUM_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    NUM_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    PACKET_RATE_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    audio_sample_rate: float
    num_channels: int
    num_samples: int
    packet_rate: float
    sample_rate: float
    def __init__(self, sample_rate: _Optional[float] = ..., num_channels: _Optional[int] = ..., num_samples: _Optional[int] = ..., packet_rate: _Optional[float] = ..., audio_sample_rate: _Optional[float] = ...) -> None: ...
