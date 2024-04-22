from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.gpu import gpu_origin_pb2 as _gpu_origin_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorsToSegmentationCalculatorOptions(_message.Message):
    __slots__ = ["activation", "gpu_origin", "output_layer_index"]
    class Activation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    GPU_ORIGIN_FIELD_NUMBER: _ClassVar[int]
    NONE: TensorsToSegmentationCalculatorOptions.Activation
    OUTPUT_LAYER_INDEX_FIELD_NUMBER: _ClassVar[int]
    SIGMOID: TensorsToSegmentationCalculatorOptions.Activation
    SOFTMAX: TensorsToSegmentationCalculatorOptions.Activation
    activation: TensorsToSegmentationCalculatorOptions.Activation
    ext: _descriptor.FieldDescriptor
    gpu_origin: _gpu_origin_pb2.GpuOrigin.Mode
    output_layer_index: int
    def __init__(self, gpu_origin: _Optional[_Union[_gpu_origin_pb2.GpuOrigin.Mode, str]] = ..., activation: _Optional[_Union[TensorsToSegmentationCalculatorOptions.Activation, str]] = ..., output_layer_index: _Optional[int] = ...) -> None: ...
