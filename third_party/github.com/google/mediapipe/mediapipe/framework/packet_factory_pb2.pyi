from google.protobuf.internal import containers as _containers
from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PacketFactoryConfig(_message.Message):
    __slots__ = ["external_output", "options", "output_side_packet", "packet_factory"]
    EXTERNAL_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SIDE_PACKET_FIELD_NUMBER: _ClassVar[int]
    PACKET_FACTORY_FIELD_NUMBER: _ClassVar[int]
    external_output: str
    options: PacketFactoryOptions
    output_side_packet: str
    packet_factory: str
    def __init__(self, packet_factory: _Optional[str] = ..., output_side_packet: _Optional[str] = ..., external_output: _Optional[str] = ..., options: _Optional[_Union[PacketFactoryOptions, _Mapping]] = ...) -> None: ...

class PacketFactoryOptions(_message.Message):
    __slots__ = []
    Extensions: _python_message._ExtensionDict
    def __init__(self) -> None: ...

class PacketManagerConfig(_message.Message):
    __slots__ = ["packet"]
    PACKET_FIELD_NUMBER: _ClassVar[int]
    packet: _containers.RepeatedCompositeFieldContainer[PacketFactoryConfig]
    def __init__(self, packet: _Optional[_Iterable[_Union[PacketFactoryConfig, _Mapping]]] = ...) -> None: ...
