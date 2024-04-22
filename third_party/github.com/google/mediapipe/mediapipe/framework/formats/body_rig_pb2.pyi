from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Joint(_message.Message):
    __slots__ = ["rotation_6d", "visibility"]
    ROTATION_6D_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    rotation_6d: _containers.RepeatedScalarFieldContainer[float]
    visibility: float
    def __init__(self, rotation_6d: _Optional[_Iterable[float]] = ..., visibility: _Optional[float] = ...) -> None: ...

class JointList(_message.Message):
    __slots__ = ["joint"]
    JOINT_FIELD_NUMBER: _ClassVar[int]
    joint: _containers.RepeatedCompositeFieldContainer[Joint]
    def __init__(self, joint: _Optional[_Iterable[_Union[Joint, _Mapping]]] = ...) -> None: ...
