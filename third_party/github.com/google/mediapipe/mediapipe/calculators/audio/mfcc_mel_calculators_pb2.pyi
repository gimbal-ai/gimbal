from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MelSpectrumCalculatorOptions(_message.Message):
    __slots__ = ["channel_count", "max_frequency_hertz", "min_frequency_hertz"]
    CHANNEL_COUNT_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MAX_FREQUENCY_HERTZ_FIELD_NUMBER: _ClassVar[int]
    MIN_FREQUENCY_HERTZ_FIELD_NUMBER: _ClassVar[int]
    channel_count: int
    ext: _descriptor.FieldDescriptor
    max_frequency_hertz: float
    min_frequency_hertz: float
    def __init__(self, channel_count: _Optional[int] = ..., min_frequency_hertz: _Optional[float] = ..., max_frequency_hertz: _Optional[float] = ...) -> None: ...

class MfccCalculatorOptions(_message.Message):
    __slots__ = ["mel_spectrum_params", "mfcc_count"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MEL_SPECTRUM_PARAMS_FIELD_NUMBER: _ClassVar[int]
    MFCC_COUNT_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    mel_spectrum_params: MelSpectrumCalculatorOptions
    mfcc_count: int
    def __init__(self, mel_spectrum_params: _Optional[_Union[MelSpectrumCalculatorOptions, _Mapping]] = ..., mfcc_count: _Optional[int] = ...) -> None: ...
