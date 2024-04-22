from gogoproto import gogo_pb2 as _gogo_pb2
from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ArgusCamSourceCalculatorOptions(_message.Message):
    __slots__ = ["device_uuid", "target_frame_rate"]
    DEVICE_UUID_FIELD_NUMBER: _ClassVar[int]
    TARGET_FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    device_uuid: str
    target_frame_rate: int
    def __init__(self, target_frame_rate: _Optional[int] = ..., device_uuid: _Optional[str] = ...) -> None: ...

class ArgusCamSourceSubgraphOptions(_message.Message):
    __slots__ = ["device_uuid", "target_frame_rate"]
    DEVICE_UUID_FIELD_NUMBER: _ClassVar[int]
    TARGET_FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    device_uuid: str
    target_frame_rate: int
    def __init__(self, target_frame_rate: _Optional[int] = ..., device_uuid: _Optional[str] = ...) -> None: ...
