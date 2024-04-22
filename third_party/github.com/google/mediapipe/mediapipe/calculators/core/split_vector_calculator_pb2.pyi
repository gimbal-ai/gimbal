from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Range(_message.Message):
    __slots__ = ["begin", "end"]
    BEGIN_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    begin: int
    end: int
    def __init__(self, begin: _Optional[int] = ..., end: _Optional[int] = ...) -> None: ...

class SplitVectorCalculatorOptions(_message.Message):
    __slots__ = ["combine_outputs", "element_only", "ranges"]
    COMBINE_OUTPUTS_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_ONLY_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    RANGES_FIELD_NUMBER: _ClassVar[int]
    combine_outputs: bool
    element_only: bool
    ext: _descriptor.FieldDescriptor
    ranges: _containers.RepeatedCompositeFieldContainer[Range]
    def __init__(self, ranges: _Optional[_Iterable[_Union[Range, _Mapping]]] = ..., element_only: bool = ..., combine_outputs: bool = ...) -> None: ...
