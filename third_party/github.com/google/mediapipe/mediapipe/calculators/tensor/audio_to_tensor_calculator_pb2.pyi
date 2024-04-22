from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AudioToTensorCalculatorOptions(_message.Message):
    __slots__ = ["check_inconsistent_timestamps", "dft_tensor_format", "fft_size", "flush_mode", "num_channels", "num_overlapping_samples", "num_samples", "padding_samples_after", "padding_samples_before", "source_sample_rate", "stream_mode", "target_sample_rate", "volume_gain_db"]
    class DftTensorFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class FlushMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    A2T_DFT_TENSOR_FORMAT_UNKNOWN: AudioToTensorCalculatorOptions.DftTensorFormat
    A2T_WITHOUT_DC_AND_NYQUIST: AudioToTensorCalculatorOptions.DftTensorFormat
    A2T_WITH_DC_AND_NYQUIST: AudioToTensorCalculatorOptions.DftTensorFormat
    A2T_WITH_NYQUIST: AudioToTensorCalculatorOptions.DftTensorFormat
    CHECK_INCONSISTENT_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    DFT_TENSOR_FORMAT_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FFT_SIZE_FIELD_NUMBER: _ClassVar[int]
    FLUSH_MODE_ENTIRE_TAIL_AT_TIMESTAMP_MAX: AudioToTensorCalculatorOptions.FlushMode
    FLUSH_MODE_FIELD_NUMBER: _ClassVar[int]
    FLUSH_MODE_NONE: AudioToTensorCalculatorOptions.FlushMode
    FLUSH_MODE_PROCEED_AS_USUAL: AudioToTensorCalculatorOptions.FlushMode
    NUM_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    NUM_OVERLAPPING_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    NUM_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    PADDING_SAMPLES_AFTER_FIELD_NUMBER: _ClassVar[int]
    PADDING_SAMPLES_BEFORE_FIELD_NUMBER: _ClassVar[int]
    SOURCE_SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    STREAM_MODE_FIELD_NUMBER: _ClassVar[int]
    TARGET_SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    VOLUME_GAIN_DB_FIELD_NUMBER: _ClassVar[int]
    check_inconsistent_timestamps: bool
    dft_tensor_format: AudioToTensorCalculatorOptions.DftTensorFormat
    ext: _descriptor.FieldDescriptor
    fft_size: int
    flush_mode: AudioToTensorCalculatorOptions.FlushMode
    num_channels: int
    num_overlapping_samples: int
    num_samples: int
    padding_samples_after: int
    padding_samples_before: int
    source_sample_rate: float
    stream_mode: bool
    target_sample_rate: float
    volume_gain_db: float
    def __init__(self, num_channels: _Optional[int] = ..., num_samples: _Optional[int] = ..., num_overlapping_samples: _Optional[int] = ..., target_sample_rate: _Optional[float] = ..., stream_mode: bool = ..., check_inconsistent_timestamps: bool = ..., fft_size: _Optional[int] = ..., padding_samples_before: _Optional[int] = ..., padding_samples_after: _Optional[int] = ..., flush_mode: _Optional[_Union[AudioToTensorCalculatorOptions.FlushMode, str]] = ..., dft_tensor_format: _Optional[_Union[AudioToTensorCalculatorOptions.DftTensorFormat, str]] = ..., volume_gain_db: _Optional[float] = ..., source_sample_rate: _Optional[float] = ...) -> None: ...
