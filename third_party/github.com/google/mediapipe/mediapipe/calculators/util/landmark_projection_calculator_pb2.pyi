from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class LandmarkProjectionCalculatorOptions(_message.Message):
    __slots__ = ["ignore_rotation"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    IGNORE_ROTATION_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    ignore_rotation: bool
    def __init__(self, ignore_rotation: bool = ...) -> None: ...
