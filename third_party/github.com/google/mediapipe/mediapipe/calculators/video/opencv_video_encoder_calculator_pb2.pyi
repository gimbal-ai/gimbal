from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OpenCvVideoEncoderCalculatorOptions(_message.Message):
    __slots__ = ["codec", "fps", "height", "video_format", "width"]
    CODEC_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FPS_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    VIDEO_FORMAT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    codec: str
    ext: _descriptor.FieldDescriptor
    fps: float
    height: int
    video_format: str
    width: int
    def __init__(self, codec: _Optional[str] = ..., video_format: _Optional[str] = ..., fps: _Optional[float] = ..., width: _Optional[int] = ..., height: _Optional[int] = ...) -> None: ...
