from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class BertPreprocessorCalculatorOptions(_message.Message):
    __slots__ = ["bert_max_seq_len", "has_dynamic_input_tensors"]
    BERT_MAX_SEQ_LEN_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    HAS_DYNAMIC_INPUT_TENSORS_FIELD_NUMBER: _ClassVar[int]
    bert_max_seq_len: int
    ext: _descriptor.FieldDescriptor
    has_dynamic_input_tensors: bool
    def __init__(self, bert_max_seq_len: _Optional[int] = ..., has_dynamic_input_tensors: bool = ...) -> None: ...
