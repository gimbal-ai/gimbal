from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FrameSelectionSolutionEvaluatorOptions(_message.Message):
    __slots__ = []
    Extensions: _python_message._ExtensionDict
    def __init__(self) -> None: ...

class FrameSelectionSolutionEvaluatorType(_message.Message):
    __slots__ = ["class_name", "options"]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    class_name: str
    options: FrameSelectionSolutionEvaluatorOptions
    def __init__(self, class_name: _Optional[str] = ..., options: _Optional[_Union[FrameSelectionSolutionEvaluatorOptions, _Mapping]] = ...) -> None: ...
