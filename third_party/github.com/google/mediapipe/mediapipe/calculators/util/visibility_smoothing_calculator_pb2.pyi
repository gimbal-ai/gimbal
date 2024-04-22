from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VisibilitySmoothingCalculatorOptions(_message.Message):
    __slots__ = ["low_pass_filter", "no_filter"]
    class LowPassFilter(_message.Message):
        __slots__ = ["alpha"]
        ALPHA_FIELD_NUMBER: _ClassVar[int]
        alpha: float
        def __init__(self, alpha: _Optional[float] = ...) -> None: ...
    class NoFilter(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    EXT_FIELD_NUMBER: _ClassVar[int]
    LOW_PASS_FILTER_FIELD_NUMBER: _ClassVar[int]
    NO_FILTER_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    low_pass_filter: VisibilitySmoothingCalculatorOptions.LowPassFilter
    no_filter: VisibilitySmoothingCalculatorOptions.NoFilter
    def __init__(self, no_filter: _Optional[_Union[VisibilitySmoothingCalculatorOptions.NoFilter, _Mapping]] = ..., low_pass_filter: _Optional[_Union[VisibilitySmoothingCalculatorOptions.LowPassFilter, _Mapping]] = ...) -> None: ...
