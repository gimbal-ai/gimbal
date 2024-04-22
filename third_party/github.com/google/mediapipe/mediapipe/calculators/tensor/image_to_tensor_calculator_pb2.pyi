from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.gpu import gpu_origin_pb2 as _gpu_origin_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ImageToTensorCalculatorOptions(_message.Message):
    __slots__ = ["border_mode", "gpu_origin", "keep_aspect_ratio", "output_tensor_float_range", "output_tensor_height", "output_tensor_int_range", "output_tensor_uint_range", "output_tensor_width"]
    class BorderMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class FloatRange(_message.Message):
        __slots__ = ["max", "min"]
        MAX_FIELD_NUMBER: _ClassVar[int]
        MIN_FIELD_NUMBER: _ClassVar[int]
        max: float
        min: float
        def __init__(self, min: _Optional[float] = ..., max: _Optional[float] = ...) -> None: ...
    class IntRange(_message.Message):
        __slots__ = ["max", "min"]
        MAX_FIELD_NUMBER: _ClassVar[int]
        MIN_FIELD_NUMBER: _ClassVar[int]
        max: int
        min: int
        def __init__(self, min: _Optional[int] = ..., max: _Optional[int] = ...) -> None: ...
    class UIntRange(_message.Message):
        __slots__ = ["max", "min"]
        MAX_FIELD_NUMBER: _ClassVar[int]
        MIN_FIELD_NUMBER: _ClassVar[int]
        max: int
        min: int
        def __init__(self, min: _Optional[int] = ..., max: _Optional[int] = ...) -> None: ...
    BORDER_MODE_FIELD_NUMBER: _ClassVar[int]
    BORDER_REPLICATE: ImageToTensorCalculatorOptions.BorderMode
    BORDER_UNSPECIFIED: ImageToTensorCalculatorOptions.BorderMode
    BORDER_ZERO: ImageToTensorCalculatorOptions.BorderMode
    EXT_FIELD_NUMBER: _ClassVar[int]
    GPU_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    KEEP_ASPECT_RATIO_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSOR_FLOAT_RANGE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSOR_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSOR_INT_RANGE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSOR_UINT_RANGE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSOR_WIDTH_FIELD_NUMBER: _ClassVar[int]
    border_mode: ImageToTensorCalculatorOptions.BorderMode
    ext: _descriptor.FieldDescriptor
    gpu_origin: _gpu_origin_pb2.GpuOrigin.Mode
    keep_aspect_ratio: bool
    output_tensor_float_range: ImageToTensorCalculatorOptions.FloatRange
    output_tensor_height: int
    output_tensor_int_range: ImageToTensorCalculatorOptions.IntRange
    output_tensor_uint_range: ImageToTensorCalculatorOptions.UIntRange
    output_tensor_width: int
    def __init__(self, output_tensor_width: _Optional[int] = ..., output_tensor_height: _Optional[int] = ..., keep_aspect_ratio: bool = ..., output_tensor_float_range: _Optional[_Union[ImageToTensorCalculatorOptions.FloatRange, _Mapping]] = ..., output_tensor_int_range: _Optional[_Union[ImageToTensorCalculatorOptions.IntRange, _Mapping]] = ..., output_tensor_uint_range: _Optional[_Union[ImageToTensorCalculatorOptions.UIntRange, _Mapping]] = ..., gpu_origin: _Optional[_Union[_gpu_origin_pb2.GpuOrigin.Mode, str]] = ..., border_mode: _Optional[_Union[ImageToTensorCalculatorOptions.BorderMode, str]] = ...) -> None: ...
