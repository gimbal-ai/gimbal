from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PacketClonerCalculatorOptions(_message.Message):
    __slots__ = ["output_only_when_all_inputs_received", "output_packets_only_when_all_inputs_received"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ONLY_WHEN_ALL_INPUTS_RECEIVED_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PACKETS_ONLY_WHEN_ALL_INPUTS_RECEIVED_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    output_only_when_all_inputs_received: bool
    output_packets_only_when_all_inputs_received: bool
    def __init__(self, output_only_when_all_inputs_received: bool = ..., output_packets_only_when_all_inputs_received: bool = ...) -> None: ...
