from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class VisibilityCopyCalculatorOptions(_message.Message):
    __slots__ = ["copy_presence", "copy_visibility"]
    COPY_PRESENCE_FIELD_NUMBER: _ClassVar[int]
    COPY_VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    copy_presence: bool
    copy_visibility: bool
    ext: _descriptor.FieldDescriptor
    def __init__(self, copy_visibility: bool = ..., copy_presence: bool = ...) -> None: ...
