from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OpenCVCamSourceCalculatorOptions(_message.Message):
    __slots__ = ["device_filename"]
    DEVICE_FILENAME_FIELD_NUMBER: _ClassVar[int]
    device_filename: str
    def __init__(self, device_filename: _Optional[str] = ...) -> None: ...
