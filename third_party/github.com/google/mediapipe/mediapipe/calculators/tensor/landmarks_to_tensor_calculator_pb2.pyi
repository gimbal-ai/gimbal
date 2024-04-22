from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LandmarksToTensorCalculatorOptions(_message.Message):
    __slots__ = ["attributes", "flatten"]
    class Attribute(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FLATTEN_FIELD_NUMBER: _ClassVar[int]
    PRESENCE: LandmarksToTensorCalculatorOptions.Attribute
    VISIBILITY: LandmarksToTensorCalculatorOptions.Attribute
    X: LandmarksToTensorCalculatorOptions.Attribute
    Y: LandmarksToTensorCalculatorOptions.Attribute
    Z: LandmarksToTensorCalculatorOptions.Attribute
    attributes: _containers.RepeatedScalarFieldContainer[LandmarksToTensorCalculatorOptions.Attribute]
    ext: _descriptor.FieldDescriptor
    flatten: bool
    def __init__(self, attributes: _Optional[_Iterable[_Union[LandmarksToTensorCalculatorOptions.Attribute, str]]] = ..., flatten: bool = ...) -> None: ...
