from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PacketFrequency(_message.Message):
    __slots__ = ["label", "packet_frequency_hz"]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    PACKET_FREQUENCY_HZ_FIELD_NUMBER: _ClassVar[int]
    label: str
    packet_frequency_hz: float
    def __init__(self, packet_frequency_hz: _Optional[float] = ..., label: _Optional[str] = ...) -> None: ...
