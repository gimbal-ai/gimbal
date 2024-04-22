from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class ImageFormat(_message.Message):
    __slots__ = []
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    FORMAT_GRAY16: ImageFormat.Format
    FORMAT_GRAY8: ImageFormat.Format
    FORMAT_LAB8: ImageFormat.Format
    FORMAT_SBGRA: ImageFormat.Format
    FORMAT_SRGB: ImageFormat.Format
    FORMAT_SRGB48: ImageFormat.Format
    FORMAT_SRGBA: ImageFormat.Format
    FORMAT_SRGBA64: ImageFormat.Format
    FORMAT_UNKNOWN: ImageFormat.Format
    FORMAT_VEC32F1: ImageFormat.Format
    FORMAT_VEC32F2: ImageFormat.Format
    FORMAT_VEC32F4: ImageFormat.Format
    FORMAT_YCBCR420P: ImageFormat.Format
    FORMAT_YCBCR420P10: ImageFormat.Format
    def __init__(self) -> None: ...
