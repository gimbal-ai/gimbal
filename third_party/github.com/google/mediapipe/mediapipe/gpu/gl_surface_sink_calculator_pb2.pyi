from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.gpu import scale_mode_pb2 as _scale_mode_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GlSurfaceSinkCalculatorOptions(_message.Message):
    __slots__ = ["frame_scale_mode"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FRAME_SCALE_MODE_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    frame_scale_mode: _scale_mode_pb2.ScaleMode.Mode
    def __init__(self, frame_scale_mode: _Optional[_Union[_scale_mode_pb2.ScaleMode.Mode, str]] = ...) -> None: ...
