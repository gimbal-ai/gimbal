from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TemplateArgument(_message.Message):
    __slots__ = ["dict", "element", "num", "str"]
    DICT_FIELD_NUMBER: _ClassVar[int]
    ELEMENT_FIELD_NUMBER: _ClassVar[int]
    NUM_FIELD_NUMBER: _ClassVar[int]
    STR_FIELD_NUMBER: _ClassVar[int]
    dict: TemplateDict
    element: _containers.RepeatedCompositeFieldContainer[TemplateArgument]
    num: float
    str: str
    def __init__(self, str: _Optional[str] = ..., num: _Optional[float] = ..., dict: _Optional[_Union[TemplateDict, _Mapping]] = ..., element: _Optional[_Iterable[_Union[TemplateArgument, _Mapping]]] = ...) -> None: ...

class TemplateDict(_message.Message):
    __slots__ = ["arg"]
    class Parameter(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: TemplateArgument
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[TemplateArgument, _Mapping]] = ...) -> None: ...
    ARG_FIELD_NUMBER: _ClassVar[int]
    arg: _containers.RepeatedCompositeFieldContainer[TemplateDict.Parameter]
    def __init__(self, arg: _Optional[_Iterable[_Union[TemplateDict.Parameter, _Mapping]]] = ...) -> None: ...
