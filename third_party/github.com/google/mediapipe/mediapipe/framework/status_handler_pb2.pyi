from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class StatusHandlerConfig(_message.Message):
    __slots__ = ["external_input", "input_side_packet", "options", "status_handler"]
    EXTERNAL_INPUT_FIELD_NUMBER: _ClassVar[int]
    INPUT_SIDE_PACKET_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    STATUS_HANDLER_FIELD_NUMBER: _ClassVar[int]
    external_input: _containers.RepeatedScalarFieldContainer[str]
    input_side_packet: _containers.RepeatedScalarFieldContainer[str]
    options: _mediapipe_options_pb2.MediaPipeOptions
    status_handler: str
    def __init__(self, status_handler: _Optional[str] = ..., input_side_packet: _Optional[_Iterable[str]] = ..., external_input: _Optional[_Iterable[str]] = ..., options: _Optional[_Union[_mediapipe_options_pb2.MediaPipeOptions, _Mapping]] = ...) -> None: ...
