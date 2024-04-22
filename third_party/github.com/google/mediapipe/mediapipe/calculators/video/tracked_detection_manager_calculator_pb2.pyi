from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util.tracking import tracked_detection_manager_config_pb2 as _tracked_detection_manager_config_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TrackedDetectionManagerCalculatorOptions(_message.Message):
    __slots__ = ["tracked_detection_manager_options"]
    EXT_FIELD_NUMBER: _ClassVar[int]
    TRACKED_DETECTION_MANAGER_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    tracked_detection_manager_options: _tracked_detection_manager_config_pb2.TrackedDetectionManagerConfig
    def __init__(self, tracked_detection_manager_options: _Optional[_Union[_tracked_detection_manager_config_pb2.TrackedDetectionManagerConfig, _Mapping]] = ...) -> None: ...
