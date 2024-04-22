from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BoundingBoxToDetectionsOptions(_message.Message):
    __slots__ = ["coordinate_format", "index_to_label"]
    class CoordinateFormat(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    COORDINATE_FORMAT_CENTER_ANCHORED: BoundingBoxToDetectionsOptions.CoordinateFormat
    COORDINATE_FORMAT_DIAG_CORNERS_YX: BoundingBoxToDetectionsOptions.CoordinateFormat
    COORDINATE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    COORDINATE_FORMAT_NORMALIZED_DIAG_CORNERS_XY: BoundingBoxToDetectionsOptions.CoordinateFormat
    COORDINATE_FORMAT_UNKNOWN: BoundingBoxToDetectionsOptions.CoordinateFormat
    INDEX_TO_LABEL_FIELD_NUMBER: _ClassVar[int]
    coordinate_format: BoundingBoxToDetectionsOptions.CoordinateFormat
    index_to_label: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, index_to_label: _Optional[_Iterable[str]] = ..., coordinate_format: _Optional[_Union[BoundingBoxToDetectionsOptions.CoordinateFormat, str]] = ...) -> None: ...
