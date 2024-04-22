from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GetVectorItemCalculatorOptions(_message.Message):
    __slots__ = ["item_index", "output_empty_on_oob"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    ITEM_INDEX_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_EMPTY_ON_OOB_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    item_index: int
    output_empty_on_oob: bool
    def __init__(self, item_index: _Optional[int] = ..., output_empty_on_oob: bool = ...) -> None: ...
