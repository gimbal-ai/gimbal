from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class GateCalculatorOptions(_message.Message):
    __slots__ = ["allow", "empty_packets_as_allow", "initial_gate_state"]
    class GateState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ALLOW_FIELD_NUMBER: _ClassVar[int]
    EMPTY_PACKETS_AS_ALLOW_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    GATE_ALLOW: GateCalculatorOptions.GateState
    GATE_DISALLOW: GateCalculatorOptions.GateState
    GATE_UNINITIALIZED: GateCalculatorOptions.GateState
    INITIAL_GATE_STATE_FIELD_NUMBER: _ClassVar[int]
    UNSPECIFIED: GateCalculatorOptions.GateState
    allow: bool
    empty_packets_as_allow: bool
    ext: _descriptor.FieldDescriptor
    initial_gate_state: GateCalculatorOptions.GateState
    def __init__(self, empty_packets_as_allow: bool = ..., allow: bool = ..., initial_gate_state: _Optional[_Union[GateCalculatorOptions.GateState, str]] = ...) -> None: ...
