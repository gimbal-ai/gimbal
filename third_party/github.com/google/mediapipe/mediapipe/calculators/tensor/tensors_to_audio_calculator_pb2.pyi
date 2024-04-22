from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorsToAudioCalculatorOptions(_message.Message):
    __slots__ = ["dft_tensor_format", "fft_size", "num_overlapping_samples", "num_samples", "volume_gain_db"]
    class DftTensorFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    DFT_TENSOR_FORMAT_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FFT_SIZE_FIELD_NUMBER: _ClassVar[int]
    NUM_OVERLAPPING_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    NUM_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    T2A_DFT_TENSOR_FORMAT_UNKNOWN: TensorsToAudioCalculatorOptions.DftTensorFormat
    T2A_WITHOUT_DC_AND_NYQUIST: TensorsToAudioCalculatorOptions.DftTensorFormat
    T2A_WITH_DC_AND_NYQUIST: TensorsToAudioCalculatorOptions.DftTensorFormat
    T2A_WITH_NYQUIST: TensorsToAudioCalculatorOptions.DftTensorFormat
    VOLUME_GAIN_DB_FIELD_NUMBER: _ClassVar[int]
    dft_tensor_format: TensorsToAudioCalculatorOptions.DftTensorFormat
    ext: _descriptor.FieldDescriptor
    fft_size: int
    num_overlapping_samples: int
    num_samples: int
    volume_gain_db: float
    def __init__(self, fft_size: _Optional[int] = ..., num_samples: _Optional[int] = ..., num_overlapping_samples: _Optional[int] = ..., dft_tensor_format: _Optional[_Union[TensorsToAudioCalculatorOptions.DftTensorFormat, str]] = ..., volume_gain_db: _Optional[float] = ...) -> None: ...
