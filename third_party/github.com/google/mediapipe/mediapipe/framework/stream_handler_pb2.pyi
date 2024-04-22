from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InputStreamHandlerConfig(_message.Message):
    __slots__ = ["input_stream_handler", "options"]
    INPUT_STREAM_HANDLER_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    input_stream_handler: str
    options: _mediapipe_options_pb2.MediaPipeOptions
    def __init__(self, input_stream_handler: _Optional[str] = ..., options: _Optional[_Union[_mediapipe_options_pb2.MediaPipeOptions, _Mapping]] = ...) -> None: ...

class OutputStreamHandlerConfig(_message.Message):
    __slots__ = ["input_side_packet", "options", "output_stream_handler"]
    INPUT_SIDE_PACKET_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_STREAM_HANDLER_FIELD_NUMBER: _ClassVar[int]
    input_side_packet: _containers.RepeatedScalarFieldContainer[str]
    options: _mediapipe_options_pb2.MediaPipeOptions
    output_stream_handler: str
    def __init__(self, output_stream_handler: _Optional[str] = ..., input_side_packet: _Optional[_Iterable[str]] = ..., options: _Optional[_Union[_mediapipe_options_pb2.MediaPipeOptions, _Mapping]] = ...) -> None: ...
