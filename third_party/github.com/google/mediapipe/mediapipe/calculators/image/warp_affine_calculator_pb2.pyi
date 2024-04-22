from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.gpu import gpu_origin_pb2 as _gpu_origin_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class WarpAffineCalculatorOptions(_message.Message):
    __slots__ = ["border_mode", "gpu_origin", "interpolation"]
    class BorderMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class Interpolation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BORDER_MODE_FIELD_NUMBER: _ClassVar[int]
    BORDER_REPLICATE: WarpAffineCalculatorOptions.BorderMode
    BORDER_UNSPECIFIED: WarpAffineCalculatorOptions.BorderMode
    BORDER_ZERO: WarpAffineCalculatorOptions.BorderMode
    EXT_FIELD_NUMBER: _ClassVar[int]
    GPU_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    INTERPOLATION_FIELD_NUMBER: _ClassVar[int]
    INTER_CUBIC: WarpAffineCalculatorOptions.Interpolation
    INTER_LINEAR: WarpAffineCalculatorOptions.Interpolation
    INTER_UNSPECIFIED: WarpAffineCalculatorOptions.Interpolation
    border_mode: WarpAffineCalculatorOptions.BorderMode
    ext: _descriptor.FieldDescriptor
    gpu_origin: _gpu_origin_pb2.GpuOrigin.Mode
    interpolation: WarpAffineCalculatorOptions.Interpolation
    def __init__(self, border_mode: _Optional[_Union[WarpAffineCalculatorOptions.BorderMode, str]] = ..., gpu_origin: _Optional[_Union[_gpu_origin_pb2.GpuOrigin.Mode, str]] = ..., interpolation: _Optional[_Union[WarpAffineCalculatorOptions.Interpolation, str]] = ...) -> None: ...
