from google.protobuf.internal import containers as _containers
from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PacketGeneratorConfig(_message.Message):
    __slots__ = ["external_input", "external_output", "input_side_packet", "options", "output_side_packet", "packet_generator"]
    EXTERNAL_INPUT_FIELD_NUMBER: _ClassVar[int]
    EXTERNAL_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    INPUT_SIDE_PACKET_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SIDE_PACKET_FIELD_NUMBER: _ClassVar[int]
    PACKET_GENERATOR_FIELD_NUMBER: _ClassVar[int]
    external_input: _containers.RepeatedScalarFieldContainer[str]
    external_output: _containers.RepeatedScalarFieldContainer[str]
    input_side_packet: _containers.RepeatedScalarFieldContainer[str]
    options: PacketGeneratorOptions
    output_side_packet: _containers.RepeatedScalarFieldContainer[str]
    packet_generator: str
    def __init__(self, packet_generator: _Optional[str] = ..., input_side_packet: _Optional[_Iterable[str]] = ..., external_input: _Optional[_Iterable[str]] = ..., output_side_packet: _Optional[_Iterable[str]] = ..., external_output: _Optional[_Iterable[str]] = ..., options: _Optional[_Union[PacketGeneratorOptions, _Mapping]] = ...) -> None: ...

class PacketGeneratorOptions(_message.Message):
    __slots__ = ["merge_fields"]
    Extensions: _python_message._ExtensionDict
    MERGE_FIELDS_FIELD_NUMBER: _ClassVar[int]
    merge_fields: bool
    def __init__(self, merge_fields: bool = ...) -> None: ...
