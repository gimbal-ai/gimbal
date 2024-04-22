from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MaskOverlayCalculatorOptions(_message.Message):
    __slots__ = ["mask_channel"]
    class MaskChannel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    EXT_FIELD_NUMBER: _ClassVar[int]
    MASK_CHANNEL_FIELD_NUMBER: _ClassVar[int]
    OVERLAY_MASK_CHANNEL_ALPHA: MaskOverlayCalculatorOptions.MaskChannel
    OVERLAY_MASK_CHANNEL_RED: MaskOverlayCalculatorOptions.MaskChannel
    OVERLAY_MASK_CHANNEL_UNKNOWN: MaskOverlayCalculatorOptions.MaskChannel
    ext: _descriptor.FieldDescriptor
    mask_channel: MaskOverlayCalculatorOptions.MaskChannel
    def __init__(self, mask_channel: _Optional[_Union[MaskOverlayCalculatorOptions.MaskChannel, str]] = ...) -> None: ...
