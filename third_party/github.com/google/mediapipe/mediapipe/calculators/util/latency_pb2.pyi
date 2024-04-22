from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PacketLatency(_message.Message):
    __slots__ = ["avg_latency_usec", "counts", "current_latency_usec", "interval_size_usec", "label", "num_intervals", "sum_latency_usec"]
    AVG_LATENCY_USEC_FIELD_NUMBER: _ClassVar[int]
    COUNTS_FIELD_NUMBER: _ClassVar[int]
    CURRENT_LATENCY_USEC_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_SIZE_USEC_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    NUM_INTERVALS_FIELD_NUMBER: _ClassVar[int]
    SUM_LATENCY_USEC_FIELD_NUMBER: _ClassVar[int]
    avg_latency_usec: int
    counts: _containers.RepeatedScalarFieldContainer[int]
    current_latency_usec: int
    interval_size_usec: int
    label: str
    num_intervals: int
    sum_latency_usec: int
    def __init__(self, current_latency_usec: _Optional[int] = ..., counts: _Optional[_Iterable[int]] = ..., num_intervals: _Optional[int] = ..., interval_size_usec: _Optional[int] = ..., avg_latency_usec: _Optional[int] = ..., label: _Optional[str] = ..., sum_latency_usec: _Optional[int] = ...) -> None: ...
