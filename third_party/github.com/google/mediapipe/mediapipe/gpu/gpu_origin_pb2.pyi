from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class GpuOrigin(_message.Message):
    __slots__ = []
    class Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ORIGIN_MODE_CONVENTIONAL: GpuOrigin.Mode
    ORIGIN_MODE_DEFAULT: GpuOrigin.Mode
    ORIGIN_MODE_TOP_LEFT: GpuOrigin.Mode
    def __init__(self) -> None: ...
