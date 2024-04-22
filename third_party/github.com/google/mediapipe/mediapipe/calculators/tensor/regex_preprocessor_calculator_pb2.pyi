from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RegexPreprocessorCalculatorOptions(_message.Message):
    __slots__ = ["max_seq_len"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MAX_SEQ_LEN_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    max_seq_len: int
    def __init__(self, max_seq_len: _Optional[int] = ...) -> None: ...
