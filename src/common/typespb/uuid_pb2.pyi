from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class UUID(_message.Message):
    __slots__ = ["high_bits", "low_bits"]
    HIGH_BITS_FIELD_NUMBER: _ClassVar[int]
    LOW_BITS_FIELD_NUMBER: _ClassVar[int]
    high_bits: int
    low_bits: int
    def __init__(self, high_bits: _Optional[int] = ..., low_bits: _Optional[int] = ...) -> None: ...
