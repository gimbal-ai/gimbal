from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class StopOnTokenSetCalculatorOptions(_message.Message):
    __slots__ = ["eos_tokens", "max_tokens_before_eos"]
    EOS_TOKENS_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKENS_BEFORE_EOS_FIELD_NUMBER: _ClassVar[int]
    eos_tokens: _containers.RepeatedScalarFieldContainer[int]
    max_tokens_before_eos: int
    def __init__(self, max_tokens_before_eos: _Optional[int] = ..., eos_tokens: _Optional[_Iterable[int]] = ...) -> None: ...
