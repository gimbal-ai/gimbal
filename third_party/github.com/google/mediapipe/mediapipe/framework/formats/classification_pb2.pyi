from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Classification(_message.Message):
    __slots__ = ["display_name", "index", "label", "score"]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    display_name: str
    index: int
    label: str
    score: float
    def __init__(self, index: _Optional[int] = ..., score: _Optional[float] = ..., label: _Optional[str] = ..., display_name: _Optional[str] = ...) -> None: ...

class ClassificationList(_message.Message):
    __slots__ = ["classification"]
    CLASSIFICATION_FIELD_NUMBER: _ClassVar[int]
    classification: _containers.RepeatedCompositeFieldContainer[Classification]
    def __init__(self, classification: _Optional[_Iterable[_Union[Classification, _Mapping]]] = ...) -> None: ...

class ClassificationListCollection(_message.Message):
    __slots__ = ["classification_list"]
    CLASSIFICATION_LIST_FIELD_NUMBER: _ClassVar[int]
    classification_list: _containers.RepeatedCompositeFieldContainer[ClassificationList]
    def __init__(self, classification_list: _Optional[_Iterable[_Union[ClassificationList, _Mapping]]] = ...) -> None: ...
