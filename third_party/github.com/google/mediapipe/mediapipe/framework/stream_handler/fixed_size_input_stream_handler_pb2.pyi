from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class FixedSizeInputStreamHandlerOptions(_message.Message):
    __slots__ = ["fixed_min_size", "target_queue_size", "trigger_queue_size"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FIXED_MIN_SIZE_FIELD_NUMBER: _ClassVar[int]
    TARGET_QUEUE_SIZE_FIELD_NUMBER: _ClassVar[int]
    TRIGGER_QUEUE_SIZE_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    fixed_min_size: bool
    target_queue_size: int
    trigger_queue_size: int
    def __init__(self, trigger_queue_size: _Optional[int] = ..., target_queue_size: _Optional[int] = ..., fixed_min_size: bool = ...) -> None: ...
