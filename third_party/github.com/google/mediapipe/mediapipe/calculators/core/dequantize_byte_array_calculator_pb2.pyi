from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class DequantizeByteArrayCalculatorOptions(_message.Message):
    __slots__ = ["max_quantized_value", "min_quantized_value"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MAX_QUANTIZED_VALUE_FIELD_NUMBER: _ClassVar[int]
    MIN_QUANTIZED_VALUE_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    max_quantized_value: float
    min_quantized_value: float
    def __init__(self, max_quantized_value: _Optional[float] = ..., min_quantized_value: _Optional[float] = ...) -> None: ...
