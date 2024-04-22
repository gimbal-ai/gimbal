from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class PacketThinnerCalculatorOptions(_message.Message):
    __slots__ = ["end_time", "period", "start_time", "sync_output_timestamps", "thinner_type", "update_frame_rate"]
    class ThinnerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ASYNC: PacketThinnerCalculatorOptions.ThinnerType
    END_TIME_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    PERIOD_FIELD_NUMBER: _ClassVar[int]
    START_TIME_FIELD_NUMBER: _ClassVar[int]
    SYNC: PacketThinnerCalculatorOptions.ThinnerType
    SYNC_OUTPUT_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    THINNER_TYPE_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN: PacketThinnerCalculatorOptions.ThinnerType
    UPDATE_FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    end_time: int
    ext: _descriptor.FieldDescriptor
    period: int
    start_time: int
    sync_output_timestamps: bool
    thinner_type: PacketThinnerCalculatorOptions.ThinnerType
    update_frame_rate: bool
    def __init__(self, thinner_type: _Optional[_Union[PacketThinnerCalculatorOptions.ThinnerType, str]] = ..., period: _Optional[int] = ..., start_time: _Optional[int] = ..., end_time: _Optional[int] = ..., sync_output_timestamps: bool = ..., update_frame_rate: bool = ...) -> None: ...
