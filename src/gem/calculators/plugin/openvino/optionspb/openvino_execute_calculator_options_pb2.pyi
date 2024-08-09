from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class OpenVinoExecuteCalculatorOptions(_message.Message):
    __slots__ = ["loopback_input_indices", "loopback_output_indices"]
    LOOPBACK_INPUT_INDICES_FIELD_NUMBER: _ClassVar[int]
    LOOPBACK_OUTPUT_INDICES_FIELD_NUMBER: _ClassVar[int]
    loopback_input_indices: _containers.RepeatedScalarFieldContainer[int]
    loopback_output_indices: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, loopback_input_indices: _Optional[_Iterable[int]] = ..., loopback_output_indices: _Optional[_Iterable[int]] = ...) -> None: ...
