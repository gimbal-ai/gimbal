from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GlContextOptions(_message.Message):
    __slots__ = ["gl_context_name"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    GL_CONTEXT_NAME_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    gl_context_name: str
    def __init__(self, gl_context_name: _Optional[str] = ...) -> None: ...
