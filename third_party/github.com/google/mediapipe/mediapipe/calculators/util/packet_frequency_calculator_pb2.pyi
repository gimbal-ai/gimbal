from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PacketFrequencyCalculatorOptions(_message.Message):
    __slots__ = ["label", "time_window_sec"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    TIME_WINDOW_SEC_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    label: _containers.RepeatedScalarFieldContainer[str]
    time_window_sec: float
    def __init__(self, time_window_sec: _Optional[float] = ..., label: _Optional[_Iterable[str]] = ...) -> None: ...
