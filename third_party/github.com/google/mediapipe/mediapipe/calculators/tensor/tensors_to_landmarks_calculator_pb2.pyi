from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorsToLandmarksCalculatorOptions(_message.Message):
    __slots__ = ["flip_horizontally", "flip_vertically", "input_image_height", "input_image_width", "normalize_z", "num_landmarks", "presence_activation", "visibility_activation"]
    class Activation(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ACTIVATION_NONE: TensorsToLandmarksCalculatorOptions.Activation
    ACTIVATION_SIGMOID: TensorsToLandmarksCalculatorOptions.Activation
    EXT_FIELD_NUMBER: _ClassVar[int]
    FLIP_HORIZONTALLY_FIELD_NUMBER: _ClassVar[int]
    FLIP_VERTICALLY_FIELD_NUMBER: _ClassVar[int]
    INPUT_IMAGE_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    INPUT_IMAGE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    NORMALIZE_Z_FIELD_NUMBER: _ClassVar[int]
    NUM_LANDMARKS_FIELD_NUMBER: _ClassVar[int]
    PRESENCE_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    VISIBILITY_ACTIVATION_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    flip_horizontally: bool
    flip_vertically: bool
    input_image_height: int
    input_image_width: int
    normalize_z: float
    num_landmarks: int
    presence_activation: TensorsToLandmarksCalculatorOptions.Activation
    visibility_activation: TensorsToLandmarksCalculatorOptions.Activation
    def __init__(self, num_landmarks: _Optional[int] = ..., input_image_width: _Optional[int] = ..., input_image_height: _Optional[int] = ..., flip_vertically: bool = ..., flip_horizontally: bool = ..., normalize_z: _Optional[float] = ..., visibility_activation: _Optional[_Union[TensorsToLandmarksCalculatorOptions.Activation, str]] = ..., presence_activation: _Optional[_Union[TensorsToLandmarksCalculatorOptions.Activation, str]] = ...) -> None: ...
