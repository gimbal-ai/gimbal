from mediapipe.framework.formats.annotation import rasterization_pb2 as _rasterization_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LocationData(_message.Message):
    __slots__ = ["bounding_box", "format", "mask", "relative_bounding_box", "relative_keypoints"]
    class Format(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class BinaryMask(_message.Message):
        __slots__ = ["height", "rasterization", "width"]
        HEIGHT_FIELD_NUMBER: _ClassVar[int]
        RASTERIZATION_FIELD_NUMBER: _ClassVar[int]
        WIDTH_FIELD_NUMBER: _ClassVar[int]
        height: int
        rasterization: _rasterization_pb2.Rasterization
        width: int
        def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., rasterization: _Optional[_Union[_rasterization_pb2.Rasterization, _Mapping]] = ...) -> None: ...
    class BoundingBox(_message.Message):
        __slots__ = ["height", "width", "xmin", "ymin"]
        HEIGHT_FIELD_NUMBER: _ClassVar[int]
        WIDTH_FIELD_NUMBER: _ClassVar[int]
        XMIN_FIELD_NUMBER: _ClassVar[int]
        YMIN_FIELD_NUMBER: _ClassVar[int]
        height: int
        width: int
        xmin: int
        ymin: int
        def __init__(self, xmin: _Optional[int] = ..., ymin: _Optional[int] = ..., width: _Optional[int] = ..., height: _Optional[int] = ...) -> None: ...
    class RelativeBoundingBox(_message.Message):
        __slots__ = ["height", "width", "xmin", "ymin"]
        HEIGHT_FIELD_NUMBER: _ClassVar[int]
        WIDTH_FIELD_NUMBER: _ClassVar[int]
        XMIN_FIELD_NUMBER: _ClassVar[int]
        YMIN_FIELD_NUMBER: _ClassVar[int]
        height: float
        width: float
        xmin: float
        ymin: float
        def __init__(self, xmin: _Optional[float] = ..., ymin: _Optional[float] = ..., width: _Optional[float] = ..., height: _Optional[float] = ...) -> None: ...
    class RelativeKeypoint(_message.Message):
        __slots__ = ["keypoint_label", "score", "x", "y"]
        KEYPOINT_LABEL_FIELD_NUMBER: _ClassVar[int]
        SCORE_FIELD_NUMBER: _ClassVar[int]
        X_FIELD_NUMBER: _ClassVar[int]
        Y_FIELD_NUMBER: _ClassVar[int]
        keypoint_label: str
        score: float
        x: float
        y: float
        def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., keypoint_label: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...
    BOUNDING_BOX_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    LOCATION_FORMAT_BOUNDING_BOX: LocationData.Format
    LOCATION_FORMAT_GLOBAL: LocationData.Format
    LOCATION_FORMAT_MASK: LocationData.Format
    LOCATION_FORMAT_RELATIVE_BOUNDING_BOX: LocationData.Format
    MASK_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_BOUNDING_BOX_FIELD_NUMBER: _ClassVar[int]
    RELATIVE_KEYPOINTS_FIELD_NUMBER: _ClassVar[int]
    bounding_box: LocationData.BoundingBox
    format: LocationData.Format
    mask: LocationData.BinaryMask
    relative_bounding_box: LocationData.RelativeBoundingBox
    relative_keypoints: _containers.RepeatedCompositeFieldContainer[LocationData.RelativeKeypoint]
    def __init__(self, format: _Optional[_Union[LocationData.Format, str]] = ..., bounding_box: _Optional[_Union[LocationData.BoundingBox, _Mapping]] = ..., relative_bounding_box: _Optional[_Union[LocationData.RelativeBoundingBox, _Mapping]] = ..., mask: _Optional[_Union[LocationData.BinaryMask, _Mapping]] = ..., relative_keypoints: _Optional[_Iterable[_Union[LocationData.RelativeKeypoint, _Mapping]]] = ...) -> None: ...
