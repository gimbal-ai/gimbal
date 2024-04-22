from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Color(_message.Message):
    __slots__ = ["b", "g", "r"]
    B_FIELD_NUMBER: _ClassVar[int]
    G_FIELD_NUMBER: _ClassVar[int]
    R_FIELD_NUMBER: _ClassVar[int]
    b: int
    g: int
    r: int
    def __init__(self, r: _Optional[int] = ..., g: _Optional[int] = ..., b: _Optional[int] = ...) -> None: ...

class ColorMap(_message.Message):
    __slots__ = ["label_to_color"]
    class LabelToColorEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Color
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Color, _Mapping]] = ...) -> None: ...
    LABEL_TO_COLOR_FIELD_NUMBER: _ClassVar[int]
    label_to_color: _containers.MessageMap[str, Color]
    def __init__(self, label_to_color: _Optional[_Mapping[str, Color]] = ...) -> None: ...
