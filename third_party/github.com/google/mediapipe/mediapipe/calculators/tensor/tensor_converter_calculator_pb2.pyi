from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.gpu import gpu_origin_pb2 as _gpu_origin_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorConverterCalculatorOptions(_message.Message):
    __slots__ = ["custom_div", "custom_sub", "flip_vertically", "gpu_origin", "max_num_channels", "output_tensor_float_range", "row_major_matrix", "use_custom_normalization", "use_quantized_tensors", "zero_center"]
    class TensorFloatRange(_message.Message):
        __slots__ = ["max", "min"]
        MAX_FIELD_NUMBER: _ClassVar[int]
        MIN_FIELD_NUMBER: _ClassVar[int]
        max: float
        min: float
        def __init__(self, min: _Optional[float] = ..., max: _Optional[float] = ...) -> None: ...
    CUSTOM_DIV_FIELD_NUMBER: _ClassVar[int]
    CUSTOM_SUB_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FLIP_VERTICALLY_FIELD_NUMBER: _ClassVar[int]
    GPU_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    MAX_NUM_CHANNELS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_TENSOR_FLOAT_RANGE_FIELD_NUMBER: _ClassVar[int]
    ROW_MAJOR_MATRIX_FIELD_NUMBER: _ClassVar[int]
    USE_CUSTOM_NORMALIZATION_FIELD_NUMBER: _ClassVar[int]
    USE_QUANTIZED_TENSORS_FIELD_NUMBER: _ClassVar[int]
    ZERO_CENTER_FIELD_NUMBER: _ClassVar[int]
    custom_div: float
    custom_sub: float
    ext: _descriptor.FieldDescriptor
    flip_vertically: bool
    gpu_origin: _gpu_origin_pb2.GpuOrigin.Mode
    max_num_channels: int
    output_tensor_float_range: TensorConverterCalculatorOptions.TensorFloatRange
    row_major_matrix: bool
    use_custom_normalization: bool
    use_quantized_tensors: bool
    zero_center: bool
    def __init__(self, zero_center: bool = ..., use_custom_normalization: bool = ..., custom_div: _Optional[float] = ..., custom_sub: _Optional[float] = ..., flip_vertically: bool = ..., gpu_origin: _Optional[_Union[_gpu_origin_pb2.GpuOrigin.Mode, str]] = ..., max_num_channels: _Optional[int] = ..., row_major_matrix: bool = ..., use_quantized_tensors: bool = ..., output_tensor_float_range: _Optional[_Union[TensorConverterCalculatorOptions.TensorFloatRange, _Mapping]] = ...) -> None: ...
