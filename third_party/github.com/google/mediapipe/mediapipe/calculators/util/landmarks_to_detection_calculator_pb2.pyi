from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class LandmarksToDetectionCalculatorOptions(_message.Message):
    __slots__ = ["selected_landmark_indices"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    SELECTED_LANDMARK_INDICES_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    selected_landmark_indices: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, selected_landmark_indices: _Optional[_Iterable[int]] = ...) -> None: ...
