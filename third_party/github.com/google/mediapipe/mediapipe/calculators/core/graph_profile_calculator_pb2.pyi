from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GraphProfileCalculatorOptions(_message.Message):
    __slots__ = ["profile_interval"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    PROFILE_INTERVAL_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    profile_interval: int
    def __init__(self, profile_interval: _Optional[int] = ...) -> None: ...
