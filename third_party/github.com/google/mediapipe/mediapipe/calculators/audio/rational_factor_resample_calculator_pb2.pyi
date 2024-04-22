from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RationalFactorResampleCalculatorOptions(_message.Message):
    __slots__ = ["check_inconsistent_timestamps", "resampler_rational_factor_options", "target_sample_rate"]
    class ResamplerRationalFactorOptions(_message.Message):
        __slots__ = ["cutoff", "kaiser_beta", "radius"]
        CUTOFF_FIELD_NUMBER: _ClassVar[int]
        KAISER_BETA_FIELD_NUMBER: _ClassVar[int]
        RADIUS_FIELD_NUMBER: _ClassVar[int]
        cutoff: float
        kaiser_beta: float
        radius: float
        def __init__(self, radius: _Optional[float] = ..., cutoff: _Optional[float] = ..., kaiser_beta: _Optional[float] = ...) -> None: ...
    CHECK_INCONSISTENT_TIMESTAMPS_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    RESAMPLER_RATIONAL_FACTOR_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    TARGET_SAMPLE_RATE_FIELD_NUMBER: _ClassVar[int]
    check_inconsistent_timestamps: bool
    ext: _descriptor.FieldDescriptor
    resampler_rational_factor_options: RationalFactorResampleCalculatorOptions.ResamplerRationalFactorOptions
    target_sample_rate: float
    def __init__(self, target_sample_rate: _Optional[float] = ..., resampler_rational_factor_options: _Optional[_Union[RationalFactorResampleCalculatorOptions.ResamplerRationalFactorOptions, _Mapping]] = ..., check_inconsistent_timestamps: bool = ...) -> None: ...
