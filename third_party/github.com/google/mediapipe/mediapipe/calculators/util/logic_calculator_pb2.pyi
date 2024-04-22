from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LogicCalculatorOptions(_message.Message):
    __slots__ = ["input_value", "negate", "op"]
    class Operation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    AND: LogicCalculatorOptions.Operation
    EXT_FIELD_NUMBER: _ClassVar[int]
    INPUT_VALUE_FIELD_NUMBER: _ClassVar[int]
    NEGATE_FIELD_NUMBER: _ClassVar[int]
    OP_FIELD_NUMBER: _ClassVar[int]
    OR: LogicCalculatorOptions.Operation
    XOR: LogicCalculatorOptions.Operation
    ext: _descriptor.FieldDescriptor
    input_value: _containers.RepeatedScalarFieldContainer[bool]
    negate: bool
    op: LogicCalculatorOptions.Operation
    def __init__(self, op: _Optional[_Union[LogicCalculatorOptions.Operation, str]] = ..., negate: bool = ..., input_value: _Optional[_Iterable[bool]] = ...) -> None: ...
