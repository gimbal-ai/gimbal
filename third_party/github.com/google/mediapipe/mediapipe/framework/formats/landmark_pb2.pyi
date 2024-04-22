from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Landmark(_message.Message):
    __slots__ = ["presence", "visibility", "x", "y", "z"]
    PRESENCE_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    presence: float
    visibility: float
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., visibility: _Optional[float] = ..., presence: _Optional[float] = ...) -> None: ...

class LandmarkList(_message.Message):
    __slots__ = ["landmark"]
    LANDMARK_FIELD_NUMBER: _ClassVar[int]
    landmark: _containers.RepeatedCompositeFieldContainer[Landmark]
    def __init__(self, landmark: _Optional[_Iterable[_Union[Landmark, _Mapping]]] = ...) -> None: ...

class LandmarkListCollection(_message.Message):
    __slots__ = ["landmark_list"]
    LANDMARK_LIST_FIELD_NUMBER: _ClassVar[int]
    landmark_list: _containers.RepeatedCompositeFieldContainer[LandmarkList]
    def __init__(self, landmark_list: _Optional[_Iterable[_Union[LandmarkList, _Mapping]]] = ...) -> None: ...

class NormalizedLandmark(_message.Message):
    __slots__ = ["presence", "visibility", "x", "y", "z"]
    PRESENCE_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_FIELD_NUMBER: _ClassVar[int]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    presence: float
    visibility: float
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., visibility: _Optional[float] = ..., presence: _Optional[float] = ...) -> None: ...

class NormalizedLandmarkList(_message.Message):
    __slots__ = ["landmark"]
    LANDMARK_FIELD_NUMBER: _ClassVar[int]
    landmark: _containers.RepeatedCompositeFieldContainer[NormalizedLandmark]
    def __init__(self, landmark: _Optional[_Iterable[_Union[NormalizedLandmark, _Mapping]]] = ...) -> None: ...

class NormalizedLandmarkListCollection(_message.Message):
    __slots__ = ["landmark_list"]
    LANDMARK_LIST_FIELD_NUMBER: _ClassVar[int]
    landmark_list: _containers.RepeatedCompositeFieldContainer[NormalizedLandmarkList]
    def __init__(self, landmark_list: _Optional[_Iterable[_Union[NormalizedLandmarkList, _Mapping]]] = ...) -> None: ...
