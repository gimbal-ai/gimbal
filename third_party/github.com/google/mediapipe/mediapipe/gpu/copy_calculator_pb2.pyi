from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CopyCalculatorOptions(_message.Message):
    __slots__ = ["rotation"]
    class Rotation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    EXT_FIELD_NUMBER: _ClassVar[int]
    ROTATION_CCW: CopyCalculatorOptions.Rotation
    ROTATION_CCW_FLIP: CopyCalculatorOptions.Rotation
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    ROTATION_NONE: CopyCalculatorOptions.Rotation
    ext: _descriptor.FieldDescriptor
    rotation: CopyCalculatorOptions.Rotation
    def __init__(self, rotation: _Optional[_Union[CopyCalculatorOptions.Rotation, str]] = ...) -> None: ...
