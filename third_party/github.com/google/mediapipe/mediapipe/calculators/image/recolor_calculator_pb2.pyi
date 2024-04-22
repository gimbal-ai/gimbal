from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import color_pb2 as _color_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class RecolorCalculatorOptions(_message.Message):
    __slots__ = ["adjust_with_luminance", "color", "invert_mask", "mask_channel"]
    class MaskChannel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ADJUST_WITH_LUMINANCE_FIELD_NUMBER: _ClassVar[int]
    COLOR_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    INVERT_MASK_FIELD_NUMBER: _ClassVar[int]
    MASK_CHANNEL_ALPHA: RecolorCalculatorOptions.MaskChannel
    MASK_CHANNEL_FIELD_NUMBER: _ClassVar[int]
    MASK_CHANNEL_RED: RecolorCalculatorOptions.MaskChannel
    MASK_CHANNEL_UNKNOWN: RecolorCalculatorOptions.MaskChannel
    adjust_with_luminance: bool
    color: _color_pb2.Color
    ext: _descriptor.FieldDescriptor
    invert_mask: bool
    mask_channel: RecolorCalculatorOptions.MaskChannel
    def __init__(self, mask_channel: _Optional[_Union[RecolorCalculatorOptions.MaskChannel, str]] = ..., color: _Optional[_Union[_color_pb2.Color, _Mapping]] = ..., invert_mask: bool = ..., adjust_with_luminance: bool = ...) -> None: ...
