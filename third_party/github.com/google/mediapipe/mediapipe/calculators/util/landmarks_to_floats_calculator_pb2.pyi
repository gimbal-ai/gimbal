from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class LandmarksToFloatsCalculatorOptions(_message.Message):
    __slots__ = ["num_dimensions"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    NUM_DIMENSIONS_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    num_dimensions: int
    def __init__(self, num_dimensions: _Optional[int] = ...) -> None: ...
