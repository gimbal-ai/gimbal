from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PacketResamplerCalculatorOptions(_message.Message):
    __slots__ = ["base_timestamp", "end_time", "flush_last_packet", "frame_rate", "jitter", "jitter_with_reflection", "output_header", "reproducible_sampling", "round_limits", "start_time"]
    class OutputHeader(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BASE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FLUSH_LAST_PACKET_FIELD_NUMBER: _ClassVar[int]
    FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    JITTER_FIELD_NUMBER: _ClassVar[int]
    JITTER_WITH_REFLECTION_FIELD_NUMBER: _ClassVar[int]
    NONE: PacketResamplerCalculatorOptions.OutputHeader
    OUTPUT_HEADER_FIELD_NUMBER: _ClassVar[int]
    PASS_HEADER: PacketResamplerCalculatorOptions.OutputHeader
    REPRODUCIBLE_SAMPLING_FIELD_NUMBER: _ClassVar[int]
    ROUND_LIMITS_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    UPDATE_VIDEO_HEADER: PacketResamplerCalculatorOptions.OutputHeader
    base_timestamp: int
    end_time: int
    ext: _descriptor.FieldDescriptor
    flush_last_packet: bool
    frame_rate: float
    jitter: float
    jitter_with_reflection: bool
    output_header: PacketResamplerCalculatorOptions.OutputHeader
    reproducible_sampling: bool
    round_limits: bool
    start_time: int
    def __init__(self, frame_rate: _Optional[float] = ..., output_header: _Optional[_Union[PacketResamplerCalculatorOptions.OutputHeader, str]] = ..., flush_last_packet: bool = ..., jitter: _Optional[float] = ..., jitter_with_reflection: bool = ..., reproducible_sampling: bool = ..., base_timestamp: _Optional[int] = ..., start_time: _Optional[int] = ..., end_time: _Optional[int] = ..., round_limits: bool = ...) -> None: ...
