from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VideoPreStreamCalculatorOptions(_message.Message):
    __slots__ = ["fps"]
    class Fps(_message.Message):
        __slots__ = ["ratio", "value"]
        class Rational32(_message.Message):
            __slots__ = ["denominator", "numerator"]
            DENOMINATOR_FIELD_NUMBER: _ClassVar[int]
            NUMERATOR_FIELD_NUMBER: _ClassVar[int]
            denominator: int
            numerator: int
            def __init__(self, numerator: _Optional[int] = ..., denominator: _Optional[int] = ...) -> None: ...
        RATIO_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        ratio: VideoPreStreamCalculatorOptions.Fps.Rational32
        value: float
        def __init__(self, value: _Optional[float] = ..., ratio: _Optional[_Union[VideoPreStreamCalculatorOptions.Fps.Rational32, _Mapping]] = ...) -> None: ...
    EXT_FIELD_NUMBER: _ClassVar[int]
    FPS_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    fps: VideoPreStreamCalculatorOptions.Fps
    def __init__(self, fps: _Optional[_Union[VideoPreStreamCalculatorOptions.Fps, _Mapping]] = ...) -> None: ...
