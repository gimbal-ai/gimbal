from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class OpenCvImageEncoderCalculatorOptions(_message.Message):
    __slots__ = ["quality"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    quality: int
    def __init__(self, quality: _Optional[int] = ...) -> None: ...

class OpenCvImageEncoderCalculatorResults(_message.Message):
    __slots__ = ["colorspace", "encoded_image", "height", "width"]
    class ColorSpace(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    COLORSPACE_FIELD_NUMBER: _ClassVar[int]
    COLOR_SPACE_GRAYSCALE: OpenCvImageEncoderCalculatorResults.ColorSpace
    COLOR_SPACE_RGB: OpenCvImageEncoderCalculatorResults.ColorSpace
    COLOR_SPACE_UNKNOWN: OpenCvImageEncoderCalculatorResults.ColorSpace
    ENCODED_IMAGE_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    colorspace: OpenCvImageEncoderCalculatorResults.ColorSpace
    encoded_image: bytes
    height: int
    width: int
    def __init__(self, encoded_image: _Optional[bytes] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., colorspace: _Optional[_Union[OpenCvImageEncoderCalculatorResults.ColorSpace, str]] = ...) -> None: ...
