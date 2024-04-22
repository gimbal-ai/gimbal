from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class ImageFileProperties(_message.Message):
    __slots__ = ["focal_length_35mm", "focal_length_mm", "focal_length_pixels", "image_height", "image_width"]
    FOCAL_LENGTH_35MM_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_MM_FIELD_NUMBER: _ClassVar[int]
    FOCAL_LENGTH_PIXELS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    focal_length_35mm: float
    focal_length_mm: float
    focal_length_pixels: float
    image_height: int
    image_width: int
    def __init__(self, image_width: _Optional[int] = ..., image_height: _Optional[int] = ..., focal_length_mm: _Optional[float] = ..., focal_length_35mm: _Optional[float] = ..., focal_length_pixels: _Optional[float] = ...) -> None: ...
