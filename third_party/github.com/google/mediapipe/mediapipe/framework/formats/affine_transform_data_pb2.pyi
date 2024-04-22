from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AffineTransformData(_message.Message):
    __slots__ = ["rotation", "scale", "shear", "translation"]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    SHEAR_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_FIELD_NUMBER: _ClassVar[int]
    rotation: float
    scale: Vector2Data
    shear: Vector2Data
    translation: Vector2Data
    def __init__(self, translation: _Optional[_Union[Vector2Data, _Mapping]] = ..., scale: _Optional[_Union[Vector2Data, _Mapping]] = ..., shear: _Optional[_Union[Vector2Data, _Mapping]] = ..., rotation: _Optional[float] = ...) -> None: ...

class Vector2Data(_message.Message):
    __slots__ = ["x", "y"]
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...
