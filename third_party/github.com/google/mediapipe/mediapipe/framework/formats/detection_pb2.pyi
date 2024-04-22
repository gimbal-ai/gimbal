from mediapipe.framework.formats import location_data_pb2 as _location_data_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Detection(_message.Message):
    __slots__ = ["associated_detections", "detection_id", "display_name", "feature_tag", "label", "label_id", "location_data", "score", "timestamp_usec", "track_id"]
    class AssociatedDetection(_message.Message):
        __slots__ = ["confidence", "id"]
        CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
        ID_FIELD_NUMBER: _ClassVar[int]
        confidence: float
        id: int
        def __init__(self, id: _Optional[int] = ..., confidence: _Optional[float] = ...) -> None: ...
    ASSOCIATED_DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    DETECTION_ID_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    FEATURE_TAG_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    LABEL_ID_FIELD_NUMBER: _ClassVar[int]
    LOCATION_DATA_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_USEC_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    associated_detections: _containers.RepeatedCompositeFieldContainer[Detection.AssociatedDetection]
    detection_id: int
    display_name: _containers.RepeatedScalarFieldContainer[str]
    feature_tag: str
    label: _containers.RepeatedScalarFieldContainer[str]
    label_id: _containers.RepeatedScalarFieldContainer[int]
    location_data: _location_data_pb2.LocationData
    score: _containers.RepeatedScalarFieldContainer[float]
    timestamp_usec: int
    track_id: str
    def __init__(self, label: _Optional[_Iterable[str]] = ..., label_id: _Optional[_Iterable[int]] = ..., score: _Optional[_Iterable[float]] = ..., location_data: _Optional[_Union[_location_data_pb2.LocationData, _Mapping]] = ..., feature_tag: _Optional[str] = ..., track_id: _Optional[str] = ..., detection_id: _Optional[int] = ..., associated_detections: _Optional[_Iterable[_Union[Detection.AssociatedDetection, _Mapping]]] = ..., display_name: _Optional[_Iterable[str]] = ..., timestamp_usec: _Optional[int] = ...) -> None: ...

class DetectionList(_message.Message):
    __slots__ = ["detection"]
    DETECTION_FIELD_NUMBER: _ClassVar[int]
    detection: _containers.RepeatedCompositeFieldContainer[Detection]
    def __init__(self, detection: _Optional[_Iterable[_Union[Detection, _Mapping]]] = ...) -> None: ...
