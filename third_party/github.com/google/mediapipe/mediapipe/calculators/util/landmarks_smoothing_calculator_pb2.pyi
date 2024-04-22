from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LandmarksSmoothingCalculatorOptions(_message.Message):
    __slots__ = ["no_filter", "one_euro_filter", "velocity_filter"]
    class NoFilter(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class OneEuroFilter(_message.Message):
        __slots__ = ["beta", "derivate_cutoff", "disable_value_scaling", "frequency", "min_allowed_object_scale", "min_cutoff"]
        BETA_FIELD_NUMBER: _ClassVar[int]
        DERIVATE_CUTOFF_FIELD_NUMBER: _ClassVar[int]
        DISABLE_VALUE_SCALING_FIELD_NUMBER: _ClassVar[int]
        FREQUENCY_FIELD_NUMBER: _ClassVar[int]
        MIN_ALLOWED_OBJECT_SCALE_FIELD_NUMBER: _ClassVar[int]
        MIN_CUTOFF_FIELD_NUMBER: _ClassVar[int]
        beta: float
        derivate_cutoff: float
        disable_value_scaling: bool
        frequency: float
        min_allowed_object_scale: float
        min_cutoff: float
        def __init__(self, frequency: _Optional[float] = ..., min_cutoff: _Optional[float] = ..., beta: _Optional[float] = ..., derivate_cutoff: _Optional[float] = ..., min_allowed_object_scale: _Optional[float] = ..., disable_value_scaling: bool = ...) -> None: ...
    class VelocityFilter(_message.Message):
        __slots__ = ["disable_value_scaling", "min_allowed_object_scale", "velocity_scale", "window_size"]
        DISABLE_VALUE_SCALING_FIELD_NUMBER: _ClassVar[int]
        MIN_ALLOWED_OBJECT_SCALE_FIELD_NUMBER: _ClassVar[int]
        VELOCITY_SCALE_FIELD_NUMBER: _ClassVar[int]
        WINDOW_SIZE_FIELD_NUMBER: _ClassVar[int]
        disable_value_scaling: bool
        min_allowed_object_scale: float
        velocity_scale: float
        window_size: int
        def __init__(self, window_size: _Optional[int] = ..., velocity_scale: _Optional[float] = ..., min_allowed_object_scale: _Optional[float] = ..., disable_value_scaling: bool = ...) -> None: ...
    EXT_FIELD_NUMBER: _ClassVar[int]
    NO_FILTER_FIELD_NUMBER: _ClassVar[int]
    ONE_EURO_FILTER_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_FILTER_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    no_filter: LandmarksSmoothingCalculatorOptions.NoFilter
    one_euro_filter: LandmarksSmoothingCalculatorOptions.OneEuroFilter
    velocity_filter: LandmarksSmoothingCalculatorOptions.VelocityFilter
    def __init__(self, no_filter: _Optional[_Union[LandmarksSmoothingCalculatorOptions.NoFilter, _Mapping]] = ..., velocity_filter: _Optional[_Union[LandmarksSmoothingCalculatorOptions.VelocityFilter, _Mapping]] = ..., one_euro_filter: _Optional[_Union[LandmarksSmoothingCalculatorOptions.OneEuroFilter, _Mapping]] = ...) -> None: ...
