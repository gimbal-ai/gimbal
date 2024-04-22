from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SpectrogramCalculatorOptions(_message.Message):
    __slots__ = ["allow_multichannel_input", "frame_duration_seconds", "frame_overlap_seconds", "output_scale", "output_type", "pad_final_packet", "use_local_timestamp", "window_type"]
    class OutputType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class WindowType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ALLOW_MULTICHANNEL_INPUT_FIELD_NUMBER: _ClassVar[int]
    COMPLEX: SpectrogramCalculatorOptions.OutputType
    DECIBELS: SpectrogramCalculatorOptions.OutputType
    EXT_FIELD_NUMBER: _ClassVar[int]
    FRAME_DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    FRAME_OVERLAP_SECONDS_FIELD_NUMBER: _ClassVar[int]
    LINEAR_MAGNITUDE: SpectrogramCalculatorOptions.OutputType
    OUTPUT_SCALE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TYPE_FIELD_NUMBER: _ClassVar[int]
    PAD_FINAL_PACKET_FIELD_NUMBER: _ClassVar[int]
    SQUARED_MAGNITUDE: SpectrogramCalculatorOptions.OutputType
    USE_LOCAL_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    WINDOW_TYPE_COSINE: SpectrogramCalculatorOptions.WindowType
    WINDOW_TYPE_FIELD_NUMBER: _ClassVar[int]
    WINDOW_TYPE_HAMMING: SpectrogramCalculatorOptions.WindowType
    WINDOW_TYPE_HANN: SpectrogramCalculatorOptions.WindowType
    WINDOW_TYPE_SQRT_HANN: SpectrogramCalculatorOptions.WindowType
    allow_multichannel_input: bool
    ext: _descriptor.FieldDescriptor
    frame_duration_seconds: float
    frame_overlap_seconds: float
    output_scale: float
    output_type: SpectrogramCalculatorOptions.OutputType
    pad_final_packet: bool
    use_local_timestamp: bool
    window_type: SpectrogramCalculatorOptions.WindowType
    def __init__(self, frame_duration_seconds: _Optional[float] = ..., frame_overlap_seconds: _Optional[float] = ..., pad_final_packet: bool = ..., output_type: _Optional[_Union[SpectrogramCalculatorOptions.OutputType, str]] = ..., allow_multichannel_input: bool = ..., window_type: _Optional[_Union[SpectrogramCalculatorOptions.WindowType, str]] = ..., output_scale: _Optional[float] = ..., use_local_timestamp: bool = ...) -> None: ...
