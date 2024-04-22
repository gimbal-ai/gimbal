from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar

DESCRIPTOR: _descriptor.FileDescriptor

class FieldDescriptorProto(_message.Message):
    __slots__ = []
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    TYPE_BOOL: FieldDescriptorProto.Type
    TYPE_BYTES: FieldDescriptorProto.Type
    TYPE_DOUBLE: FieldDescriptorProto.Type
    TYPE_ENUM: FieldDescriptorProto.Type
    TYPE_FIXED32: FieldDescriptorProto.Type
    TYPE_FIXED64: FieldDescriptorProto.Type
    TYPE_FLOAT: FieldDescriptorProto.Type
    TYPE_GROUP: FieldDescriptorProto.Type
    TYPE_INT32: FieldDescriptorProto.Type
    TYPE_INT64: FieldDescriptorProto.Type
    TYPE_INVALID: FieldDescriptorProto.Type
    TYPE_MESSAGE: FieldDescriptorProto.Type
    TYPE_SFIXED32: FieldDescriptorProto.Type
    TYPE_SFIXED64: FieldDescriptorProto.Type
    TYPE_SINT32: FieldDescriptorProto.Type
    TYPE_SINT64: FieldDescriptorProto.Type
    TYPE_STRING: FieldDescriptorProto.Type
    TYPE_UINT32: FieldDescriptorProto.Type
    TYPE_UINT64: FieldDescriptorProto.Type
    def __init__(self) -> None: ...
