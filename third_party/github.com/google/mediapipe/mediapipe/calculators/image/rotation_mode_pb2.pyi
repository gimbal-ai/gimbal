from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class RotationMode(_message.Message):
    __slots__ = []
    class Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ROTATION_MODE_ROTATION_0: RotationMode.Mode
    ROTATION_MODE_ROTATION_180: RotationMode.Mode
    ROTATION_MODE_ROTATION_270: RotationMode.Mode
    ROTATION_MODE_ROTATION_90: RotationMode.Mode
    ROTATION_MODE_UNKNOWN: RotationMode.Mode
    def __init__(self) -> None: ...
