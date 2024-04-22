from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FlowLimiterCalculatorOptions(_message.Message):
    __slots__ = ["in_flight_timeout", "max_in_flight", "max_in_queue"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    IN_FLIGHT_TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    MAX_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
    MAX_IN_QUEUE_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    in_flight_timeout: int
    max_in_flight: int
    max_in_queue: int
    def __init__(self, max_in_flight: _Optional[int] = ..., max_in_queue: _Optional[int] = ..., in_flight_timeout: _Optional[int] = ...) -> None: ...
