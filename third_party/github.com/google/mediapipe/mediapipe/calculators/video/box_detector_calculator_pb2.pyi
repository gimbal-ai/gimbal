from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util.tracking import box_detector_pb2 as _box_detector_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BoxDetectorCalculatorOptions(_message.Message):
    __slots__ = ["detector_options", "index_proto_filename"]
    DETECTOR_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    INDEX_PROTO_FILENAME_FIELD_NUMBER: _ClassVar[int]
    detector_options: _box_detector_pb2.BoxDetectorOptions
    ext: _descriptor.FieldDescriptor
    index_proto_filename: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, detector_options: _Optional[_Union[_box_detector_pb2.BoxDetectorOptions, _Mapping]] = ..., index_proto_filename: _Optional[_Iterable[str]] = ...) -> None: ...
