from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class CalculatorOptions(_message.Message):
    __slots__ = ["merge_fields"]
    Extensions: _python_message._ExtensionDict
    MERGE_FIELDS_FIELD_NUMBER: _ClassVar[int]
    merge_fields: bool
    def __init__(self, merge_fields: bool = ...) -> None: ...
