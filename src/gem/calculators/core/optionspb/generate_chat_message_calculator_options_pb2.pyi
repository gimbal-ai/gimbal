from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class GenerateChatMessageCalculatorOptions(_message.Message):
    __slots__ = ["add_generation_prompt", "message_template", "preset_system_prompt"]
    ADD_GENERATION_PROMPT_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_TEMPLATE_FIELD_NUMBER: _ClassVar[int]
    PRESET_SYSTEM_PROMPT_FIELD_NUMBER: _ClassVar[int]
    add_generation_prompt: bool
    message_template: str
    preset_system_prompt: str
    def __init__(self, message_template: _Optional[str] = ..., preset_system_prompt: _Optional[str] = ..., add_generation_prompt: bool = ...) -> None: ...
