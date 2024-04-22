from gogoproto import gogo_pb2 as _gogo_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor
IMAGE_COLOR_CHANNEL_BLUE: ImageColorChannel
IMAGE_COLOR_CHANNEL_GRAY: ImageColorChannel
IMAGE_COLOR_CHANNEL_GREEN: ImageColorChannel
IMAGE_COLOR_CHANNEL_RED: ImageColorChannel
IMAGE_COLOR_CHANNEL_UNKNOWN: ImageColorChannel

class Detection(_message.Message):
    __slots__ = ["bounding_box", "label"]
    BOUNDING_BOX_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    bounding_box: NormalizedCenterRect
    label: _containers.RepeatedCompositeFieldContainer[Label]
    def __init__(self, label: _Optional[_Iterable[_Union[Label, _Mapping]]] = ..., bounding_box: _Optional[_Union[NormalizedCenterRect, _Mapping]] = ...) -> None: ...

class DetectionList(_message.Message):
    __slots__ = ["detection"]
    DETECTION_FIELD_NUMBER: _ClassVar[int]
    detection: _containers.RepeatedCompositeFieldContainer[Detection]
    def __init__(self, detection: _Optional[_Iterable[_Union[Detection, _Mapping]]] = ...) -> None: ...

class H264Chunk(_message.Message):
    __slots__ = ["eof", "frame_ts", "nal_data"]
    EOF_FIELD_NUMBER: _ClassVar[int]
    FRAME_TS_FIELD_NUMBER: _ClassVar[int]
    NAL_DATA_FIELD_NUMBER: _ClassVar[int]
    eof: bool
    frame_ts: int
    nal_data: bytes
    def __init__(self, frame_ts: _Optional[int] = ..., eof: bool = ..., nal_data: _Optional[bytes] = ...) -> None: ...

class ImageHistogram(_message.Message):
    __slots__ = ["bucket", "channel", "max", "min", "num", "sum"]
    BUCKET_FIELD_NUMBER: _ClassVar[int]
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    NUM_FIELD_NUMBER: _ClassVar[int]
    SUM_FIELD_NUMBER: _ClassVar[int]
    bucket: _containers.RepeatedScalarFieldContainer[int]
    channel: ImageColorChannel
    max: float
    min: float
    num: int
    sum: float
    def __init__(self, channel: _Optional[_Union[ImageColorChannel, str]] = ..., min: _Optional[float] = ..., max: _Optional[float] = ..., num: _Optional[int] = ..., sum: _Optional[float] = ..., bucket: _Optional[_Iterable[int]] = ...) -> None: ...

class ImageHistogramBatch(_message.Message):
    __slots__ = ["histograms"]
    HISTOGRAMS_FIELD_NUMBER: _ClassVar[int]
    histograms: _containers.RepeatedCompositeFieldContainer[ImageHistogram]
    def __init__(self, histograms: _Optional[_Iterable[_Union[ImageHistogram, _Mapping]]] = ...) -> None: ...

class ImageOverlayChunk(_message.Message):
    __slots__ = ["detections", "eof", "frame_ts", "histograms", "image_quality"]
    DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    EOF_FIELD_NUMBER: _ClassVar[int]
    FRAME_TS_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAMS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_QUALITY_FIELD_NUMBER: _ClassVar[int]
    detections: DetectionList
    eof: bool
    frame_ts: int
    histograms: ImageHistogramBatch
    image_quality: ImageQualityMetrics
    def __init__(self, frame_ts: _Optional[int] = ..., eof: bool = ..., detections: _Optional[_Union[DetectionList, _Mapping]] = ..., histograms: _Optional[_Union[ImageHistogramBatch, _Mapping]] = ..., image_quality: _Optional[_Union[ImageQualityMetrics, _Mapping]] = ...) -> None: ...

class ImageQualityMetrics(_message.Message):
    __slots__ = ["blurriness_score", "brisque_score"]
    BLURRINESS_SCORE_FIELD_NUMBER: _ClassVar[int]
    BRISQUE_SCORE_FIELD_NUMBER: _ClassVar[int]
    blurriness_score: float
    brisque_score: float
    def __init__(self, brisque_score: _Optional[float] = ..., blurriness_score: _Optional[float] = ...) -> None: ...

class Label(_message.Message):
    __slots__ = ["label", "score"]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    label: str
    score: float
    def __init__(self, label: _Optional[str] = ..., score: _Optional[float] = ...) -> None: ...

class NormalizedCenterRect(_message.Message):
    __slots__ = ["height", "width", "xc", "yc"]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    XC_FIELD_NUMBER: _ClassVar[int]
    YC_FIELD_NUMBER: _ClassVar[int]
    height: float
    width: float
    xc: float
    yc: float
    def __init__(self, xc: _Optional[float] = ..., yc: _Optional[float] = ..., width: _Optional[float] = ..., height: _Optional[float] = ...) -> None: ...

class VideoHeader(_message.Message):
    __slots__ = ["frame_rate", "height", "width"]
    FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    frame_rate: float
    height: int
    width: int
    def __init__(self, width: _Optional[int] = ..., height: _Optional[int] = ..., frame_rate: _Optional[float] = ...) -> None: ...

class ImageColorChannel(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
