from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class AssociationCalculatorOptions(_message.Message):
    __slots__ = ["min_similarity_threshold"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MIN_SIMILARITY_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    min_similarity_threshold: float
    def __init__(self, min_similarity_threshold: _Optional[float] = ...) -> None: ...
