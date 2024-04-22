from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SequenceShiftCalculatorOptions(_message.Message):
    __slots__ = ["emit_empty_packets_before_first_packet", "packet_offset"]
    EMIT_EMPTY_PACKETS_BEFORE_FIRST_PACKET_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    PACKET_OFFSET_FIELD_NUMBER: _ClassVar[int]
    emit_empty_packets_before_first_packet: bool
    ext: _descriptor.FieldDescriptor
    packet_offset: int
    def __init__(self, packet_offset: _Optional[int] = ..., emit_empty_packets_before_first_packet: bool = ...) -> None: ...
