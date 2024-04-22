from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TimeSeriesFramerCalculatorOptions(_message.Message):
    __slots__ = ["emulate_fractional_frame_overlap", "frame_duration_seconds", "frame_overlap_seconds", "pad_final_packet", "use_local_timestamp", "window_function"]
    class WindowFunction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    EMULATE_FRACTIONAL_FRAME_OVERLAP_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FRAME_DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    FRAME_OVERLAP_SECONDS_FIELD_NUMBER: _ClassVar[int]
    PAD_FINAL_PACKET_FIELD_NUMBER: _ClassVar[int]
    USE_LOCAL_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FUNCTION_FIELD_NUMBER: _ClassVar[int]
    WINDOW_HAMMING: TimeSeriesFramerCalculatorOptions.WindowFunction
    WINDOW_HANN: TimeSeriesFramerCalculatorOptions.WindowFunction
    WINDOW_NONE: TimeSeriesFramerCalculatorOptions.WindowFunction
    emulate_fractional_frame_overlap: bool
    ext: _descriptor.FieldDescriptor
    frame_duration_seconds: float
    frame_overlap_seconds: float
    pad_final_packet: bool
    use_local_timestamp: bool
    window_function: TimeSeriesFramerCalculatorOptions.WindowFunction
    def __init__(self, frame_duration_seconds: _Optional[float] = ..., frame_overlap_seconds: _Optional[float] = ..., emulate_fractional_frame_overlap: bool = ..., pad_final_packet: bool = ..., window_function: _Optional[_Union[TimeSeriesFramerCalculatorOptions.WindowFunction, str]] = ..., use_local_timestamp: bool = ...) -> None: ...
