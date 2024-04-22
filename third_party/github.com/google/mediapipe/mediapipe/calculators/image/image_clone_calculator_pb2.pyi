from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ImageCloneCalculatorOptions(_message.Message):
    __slots__ = ["output_on_gpu"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ON_GPU_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    output_on_gpu: bool
    def __init__(self, output_on_gpu: bool = ...) -> None: ...
