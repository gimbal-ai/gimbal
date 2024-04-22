from mediapipe.framework.formats.annotation import rasterization_pb2 as _rasterization_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BoundingBox(_message.Message):
    __slots__ = ["left_x", "lower_y", "right_x", "upper_y"]
    LEFT_X_FIELD_NUMBER: _ClassVar[int]
    LOWER_Y_FIELD_NUMBER: _ClassVar[int]
    RIGHT_X_FIELD_NUMBER: _ClassVar[int]
    UPPER_Y_FIELD_NUMBER: _ClassVar[int]
    left_x: int
    lower_y: int
    right_x: int
    upper_y: int
    def __init__(self, left_x: _Optional[int] = ..., upper_y: _Optional[int] = ..., right_x: _Optional[int] = ..., lower_y: _Optional[int] = ...) -> None: ...

class Locus(_message.Message):
    __slots__ = ["bounding_box", "component_locus", "concatenatable", "locus_id", "locus_id_seed", "locus_type", "region", "timestamp"]
    class LocusType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    BOUNDING_BOX_FIELD_NUMBER: _ClassVar[int]
    COMPONENT_LOCUS_FIELD_NUMBER: _ClassVar[int]
    CONCATENATABLE_FIELD_NUMBER: _ClassVar[int]
    LOCUS_ID_FIELD_NUMBER: _ClassVar[int]
    LOCUS_ID_SEED_FIELD_NUMBER: _ClassVar[int]
    LOCUS_TYPE_BOUNDING_BOX: Locus.LocusType
    LOCUS_TYPE_FIELD_NUMBER: _ClassVar[int]
    LOCUS_TYPE_GLOBAL: Locus.LocusType
    LOCUS_TYPE_REGION: Locus.LocusType
    LOCUS_TYPE_UNKNOWN: Locus.LocusType
    LOCUS_TYPE_VIDEO_TUBE: Locus.LocusType
    REGION_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    bounding_box: BoundingBox
    component_locus: _containers.RepeatedCompositeFieldContainer[Locus]
    concatenatable: bool
    locus_id: int
    locus_id_seed: int
    locus_type: Locus.LocusType
    region: _rasterization_pb2.Rasterization
    timestamp: int
    def __init__(self, locus_type: _Optional[_Union[Locus.LocusType, str]] = ..., locus_id: _Optional[int] = ..., locus_id_seed: _Optional[int] = ..., concatenatable: bool = ..., bounding_box: _Optional[_Union[BoundingBox, _Mapping]] = ..., timestamp: _Optional[int] = ..., region: _Optional[_Union[_rasterization_pb2.Rasterization, _Mapping]] = ..., component_locus: _Optional[_Iterable[_Union[Locus, _Mapping]]] = ...) -> None: ...
