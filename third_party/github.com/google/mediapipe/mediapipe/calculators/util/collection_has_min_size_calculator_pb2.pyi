from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class CollectionHasMinSizeCalculatorOptions(_message.Message):
    __slots__ = ["min_size"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MIN_SIZE_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    min_size: int
    def __init__(self, min_size: _Optional[int] = ...) -> None: ...
