from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class ScaleMode(_message.Message):
    __slots__ = []
    class Mode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    SCALE_MODE_DEFAULT: ScaleMode.Mode
    SCALE_MODE_FILL_AND_CROP: ScaleMode.Mode
    SCALE_MODE_FIT: ScaleMode.Mode
    SCALE_MODE_STRETCH: ScaleMode.Mode
    def __init__(self) -> None: ...
