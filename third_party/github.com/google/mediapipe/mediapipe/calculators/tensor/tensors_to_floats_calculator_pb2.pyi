from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorsToFloatsCalculatorOptions(_message.Message):
    __slots__ = ["activation"]
    class Activation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    T2F_ACTIVATION_NONE: TensorsToFloatsCalculatorOptions.Activation
    T2F_ACTIVATION_SIGMOID: TensorsToFloatsCalculatorOptions.Activation
    activation: TensorsToFloatsCalculatorOptions.Activation
    ext: _descriptor.FieldDescriptor
    def __init__(self, activation: _Optional[_Union[TensorsToFloatsCalculatorOptions.Activation, str]] = ...) -> None: ...
