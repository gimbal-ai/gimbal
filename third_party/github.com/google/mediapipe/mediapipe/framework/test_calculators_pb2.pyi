from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class RandomMatrixCalculatorOptions(_message.Message):
    __slots__ = ["cols", "limit_timestamp", "rows", "start_timestamp", "timestamp_step"]
    COLS_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    LIMIT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    START_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_STEP_FIELD_NUMBER: _ClassVar[int]
    cols: int
    ext: _descriptor.FieldDescriptor
    limit_timestamp: int
    rows: int
    start_timestamp: int
    timestamp_step: int
    def __init__(self, rows: _Optional[int] = ..., cols: _Optional[int] = ..., start_timestamp: _Optional[int] = ..., limit_timestamp: _Optional[int] = ..., timestamp_step: _Optional[int] = ...) -> None: ...
