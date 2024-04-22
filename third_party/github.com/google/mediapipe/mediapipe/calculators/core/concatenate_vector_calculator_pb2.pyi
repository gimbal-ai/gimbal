from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ConcatenateVectorCalculatorOptions(_message.Message):
    __slots__ = ["only_emit_if_all_present"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    ONLY_EMIT_IF_ALL_PRESENT_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    only_emit_if_all_present: bool
    def __init__(self, only_emit_if_all_present: bool = ...) -> None: ...
