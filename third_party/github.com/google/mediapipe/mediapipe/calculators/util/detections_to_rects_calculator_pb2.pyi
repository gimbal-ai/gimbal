from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DetectionsToRectsCalculatorOptions(_message.Message):
    __slots__ = ["conversion_mode", "output_zero_rect_for_empty_detections", "rotation_vector_end_keypoint_index", "rotation_vector_start_keypoint_index", "rotation_vector_target_angle", "rotation_vector_target_angle_degrees"]
    class ConversionMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    CONVERSION_MODE_FIELD_NUMBER: _ClassVar[int]
    DEFAULT: DetectionsToRectsCalculatorOptions.ConversionMode
    EXT_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_ZERO_RECT_FOR_EMPTY_DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    ROTATION_VECTOR_END_KEYPOINT_INDEX_FIELD_NUMBER: _ClassVar[int]
    ROTATION_VECTOR_START_KEYPOINT_INDEX_FIELD_NUMBER: _ClassVar[int]
    ROTATION_VECTOR_TARGET_ANGLE_DEGREES_FIELD_NUMBER: _ClassVar[int]
    ROTATION_VECTOR_TARGET_ANGLE_FIELD_NUMBER: _ClassVar[int]
    USE_BOUNDING_BOX: DetectionsToRectsCalculatorOptions.ConversionMode
    USE_KEYPOINTS: DetectionsToRectsCalculatorOptions.ConversionMode
    conversion_mode: DetectionsToRectsCalculatorOptions.ConversionMode
    ext: _descriptor.FieldDescriptor
    output_zero_rect_for_empty_detections: bool
    rotation_vector_end_keypoint_index: int
    rotation_vector_start_keypoint_index: int
    rotation_vector_target_angle: float
    rotation_vector_target_angle_degrees: float
    def __init__(self, rotation_vector_start_keypoint_index: _Optional[int] = ..., rotation_vector_end_keypoint_index: _Optional[int] = ..., rotation_vector_target_angle: _Optional[float] = ..., rotation_vector_target_angle_degrees: _Optional[float] = ..., output_zero_rect_for_empty_detections: bool = ..., conversion_mode: _Optional[_Union[DetectionsToRectsCalculatorOptions.ConversionMode, str]] = ...) -> None: ...
