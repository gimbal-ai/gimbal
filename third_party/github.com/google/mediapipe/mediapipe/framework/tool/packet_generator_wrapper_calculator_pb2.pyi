from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.framework import packet_generator_pb2 as _packet_generator_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PacketGeneratorWrapperCalculatorOptions(_message.Message):
    __slots__ = ["options", "package", "packet_generator"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    PACKET_GENERATOR_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    options: _packet_generator_pb2.PacketGeneratorOptions
    package: str
    packet_generator: str
    def __init__(self, packet_generator: _Optional[str] = ..., options: _Optional[_Union[_packet_generator_pb2.PacketGeneratorOptions, _Mapping]] = ..., package: _Optional[str] = ...) -> None: ...
