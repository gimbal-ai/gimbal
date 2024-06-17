from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SegmentationMasksToProtoOptions(_message.Message):
    __slots__ = ["index_to_label"]
    INDEX_TO_LABEL_FIELD_NUMBER: _ClassVar[int]
    index_to_label: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, index_to_label: _Optional[_Iterable[str]] = ...) -> None: ...
