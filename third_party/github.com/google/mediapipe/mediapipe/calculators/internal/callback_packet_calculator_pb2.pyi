from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CallbackPacketCalculatorOptions(_message.Message):
    __slots__ = ["pointer", "type"]
    class PointerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    EXT_FIELD_NUMBER: _ClassVar[int]
    POINTER_FIELD_NUMBER: _ClassVar[int]
    POST_STREAM_PACKET: CallbackPacketCalculatorOptions.PointerType
    TYPE_FIELD_NUMBER: _ClassVar[int]
    UNKNOWN: CallbackPacketCalculatorOptions.PointerType
    VECTOR_PACKET: CallbackPacketCalculatorOptions.PointerType
    ext: _descriptor.FieldDescriptor
    pointer: bytes
    type: CallbackPacketCalculatorOptions.PointerType
    def __init__(self, type: _Optional[_Union[CallbackPacketCalculatorOptions.PointerType, str]] = ..., pointer: _Optional[bytes] = ...) -> None: ...
