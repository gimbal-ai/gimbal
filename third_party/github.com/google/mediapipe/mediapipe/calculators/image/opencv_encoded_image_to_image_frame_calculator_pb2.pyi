from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OpenCvEncodedImageToImageFrameCalculatorOptions(_message.Message):
    __slots__ = ["apply_orientation_from_exif_data"]
    APPLY_ORIENTATION_FROM_EXIF_DATA_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    apply_orientation_from_exif_data: bool
    ext: _descriptor.FieldDescriptor
    def __init__(self, apply_orientation_from_exif_data: bool = ...) -> None: ...
