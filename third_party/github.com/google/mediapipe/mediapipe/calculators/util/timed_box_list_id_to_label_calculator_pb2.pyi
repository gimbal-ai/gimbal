from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TimedBoxListIdToLabelCalculatorOptions(_message.Message):
    __slots__ = ["label_map_path"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    LABEL_MAP_PATH_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    label_map_path: str
    def __init__(self, label_map_path: _Optional[str] = ...) -> None: ...
