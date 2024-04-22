from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class CalculatorContractTestOptions(_message.Message):
    __slots__ = ["test_field"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    TEST_FIELD_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    test_field: float
    def __init__(self, test_field: _Optional[float] = ...) -> None: ...
