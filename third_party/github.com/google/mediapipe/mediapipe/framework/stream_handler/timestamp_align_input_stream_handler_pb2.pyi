from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class TimestampAlignInputStreamHandlerOptions(_message.Message):
    __slots__ = ["timestamp_base_tag_index"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_BASE_TAG_INDEX_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    timestamp_base_tag_index: str
    def __init__(self, timestamp_base_tag_index: _Optional[str] = ...) -> None: ...
