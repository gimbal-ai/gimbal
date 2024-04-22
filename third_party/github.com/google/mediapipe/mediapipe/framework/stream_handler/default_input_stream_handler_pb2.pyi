from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class DefaultInputStreamHandlerOptions(_message.Message):
    __slots__ = ["batch_size"]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    batch_size: int
    ext: _descriptor.FieldDescriptor
    def __init__(self, batch_size: _Optional[int] = ...) -> None: ...
