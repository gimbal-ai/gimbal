from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FeedbackTensorsCalculatorOptions(_message.Message):
    __slots__ = ["feedback_tensor_shape", "location", "num_feedback_tensors"]
    class FeedbackTensorsLocation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class TensorShape(_message.Message):
        __slots__ = ["dims"]
        DIMS_FIELD_NUMBER: _ClassVar[int]
        dims: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, dims: _Optional[_Iterable[int]] = ...) -> None: ...
    EXT_FIELD_NUMBER: _ClassVar[int]
    FEEDBACK_APPENDED: FeedbackTensorsCalculatorOptions.FeedbackTensorsLocation
    FEEDBACK_NONE: FeedbackTensorsCalculatorOptions.FeedbackTensorsLocation
    FEEDBACK_PREPENDED: FeedbackTensorsCalculatorOptions.FeedbackTensorsLocation
    FEEDBACK_TENSOR_SHAPE_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FIELD_NUMBER: _ClassVar[int]
    NUM_FEEDBACK_TENSORS_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    feedback_tensor_shape: FeedbackTensorsCalculatorOptions.TensorShape
    location: FeedbackTensorsCalculatorOptions.FeedbackTensorsLocation
    num_feedback_tensors: int
    def __init__(self, feedback_tensor_shape: _Optional[_Union[FeedbackTensorsCalculatorOptions.TensorShape, _Mapping]] = ..., num_feedback_tensors: _Optional[int] = ..., location: _Optional[_Union[FeedbackTensorsCalculatorOptions.FeedbackTensorsLocation, str]] = ...) -> None: ...
