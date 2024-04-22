from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OpenCVCamSourceCalculatorOptions(_message.Message):
    __slots__ = ["device_filename", "max_num_frames"]
    DEVICE_FILENAME_FIELD_NUMBER: _ClassVar[int]
    MAX_NUM_FRAMES_FIELD_NUMBER: _ClassVar[int]
    device_filename: str
    max_num_frames: int
    def __init__(self, device_filename: _Optional[str] = ..., max_num_frames: _Optional[int] = ...) -> None: ...

class OpenCVCamSourceSubgraphOptions(_message.Message):
    __slots__ = ["device_filename"]
    DEVICE_FILENAME_FIELD_NUMBER: _ClassVar[int]
    device_filename: str
    def __init__(self, device_filename: _Optional[str] = ...) -> None: ...
