from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SyncSetInputStreamHandlerOptions(_message.Message):
    __slots__ = ["sync_set"]
    class SyncSet(_message.Message):
        __slots__ = ["tag_index"]
        TAG_INDEX_FIELD_NUMBER: _ClassVar[int]
        tag_index: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, tag_index: _Optional[_Iterable[str]] = ...) -> None: ...
    EXT_FIELD_NUMBER: _ClassVar[int]
    SYNC_SET_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    sync_set: _containers.RepeatedCompositeFieldContainer[SyncSetInputStreamHandlerOptions.SyncSet]
    def __init__(self, sync_set: _Optional[_Iterable[_Union[SyncSetInputStreamHandlerOptions.SyncSet, _Mapping]]] = ...) -> None: ...
