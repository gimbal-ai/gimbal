from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SwitchContainerOptions(_message.Message):
    __slots__ = ["async_selection", "contained_node", "enable", "select", "synchronize_io", "tick_input_stream"]
    ASYNC_SELECTION_FIELD_NUMBER: _ClassVar[int]
    CONTAINED_NODE_FIELD_NUMBER: _ClassVar[int]
    ENABLE_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    SELECT_FIELD_NUMBER: _ClassVar[int]
    SYNCHRONIZE_IO_FIELD_NUMBER: _ClassVar[int]
    TICK_INPUT_STREAM_FIELD_NUMBER: _ClassVar[int]
    async_selection: bool
    contained_node: _containers.RepeatedCompositeFieldContainer[_calculator_pb2.CalculatorGraphConfig.Node]
    enable: bool
    ext: _descriptor.FieldDescriptor
    select: int
    synchronize_io: bool
    tick_input_stream: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, contained_node: _Optional[_Iterable[_Union[_calculator_pb2.CalculatorGraphConfig.Node, _Mapping]]] = ..., select: _Optional[int] = ..., enable: bool = ..., synchronize_io: bool = ..., async_selection: bool = ..., tick_input_stream: _Optional[_Iterable[str]] = ...) -> None: ...
