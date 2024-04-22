from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PacketLatencyCalculatorOptions(_message.Message):
    __slots__ = ["interval_size_usec", "num_intervals", "packet_labels", "reset_duration_usec"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_SIZE_USEC_FIELD_NUMBER: _ClassVar[int]
    NUM_INTERVALS_FIELD_NUMBER: _ClassVar[int]
    PACKET_LABELS_FIELD_NUMBER: _ClassVar[int]
    RESET_DURATION_USEC_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    interval_size_usec: int
    num_intervals: int
    packet_labels: _containers.RepeatedScalarFieldContainer[str]
    reset_duration_usec: int
    def __init__(self, num_intervals: _Optional[int] = ..., interval_size_usec: _Optional[int] = ..., reset_duration_usec: _Optional[int] = ..., packet_labels: _Optional[_Iterable[str]] = ...) -> None: ...
