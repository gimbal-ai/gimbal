from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.framework.formats import image_format_pb2 as _image_format_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ScaleImageCalculatorOptions(_message.Message):
    __slots__ = ["OBSOLETE_skip_linear_rgb_conversion", "algorithm", "alignment_boundary", "input_format", "max_aspect_ratio", "min_aspect_ratio", "output_format", "post_sharpening_coefficient", "preserve_aspect_ratio", "scale_to_multiple_of", "set_alignment_padding", "target_height", "target_max_area", "target_width", "use_bt709"]
    class ScaleAlgorithm(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ALGORITHM_FIELD_NUMBER: _ClassVar[int]
    ALIGNMENT_BOUNDARY_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    INPUT_FORMAT_FIELD_NUMBER: _ClassVar[int]
    MAX_ASPECT_RATIO_FIELD_NUMBER: _ClassVar[int]
    MIN_ASPECT_RATIO_FIELD_NUMBER: _ClassVar[int]
    OBSOLETE_SKIP_LINEAR_RGB_CONVERSION_FIELD_NUMBER: _ClassVar[int]
    OBSOLETE_skip_linear_rgb_conversion: bool
    OUTPUT_FORMAT_FIELD_NUMBER: _ClassVar[int]
    POST_SHARPENING_COEFFICIENT_FIELD_NUMBER: _ClassVar[int]
    PRESERVE_ASPECT_RATIO_FIELD_NUMBER: _ClassVar[int]
    SCALE_ALGO_AREA: ScaleImageCalculatorOptions.ScaleAlgorithm
    SCALE_ALGO_CUBIC: ScaleImageCalculatorOptions.ScaleAlgorithm
    SCALE_ALGO_DEFAULT: ScaleImageCalculatorOptions.ScaleAlgorithm
    SCALE_ALGO_DEFAULT_WITHOUT_UPSCALE: ScaleImageCalculatorOptions.ScaleAlgorithm
    SCALE_ALGO_LANCZOS: ScaleImageCalculatorOptions.ScaleAlgorithm
    SCALE_ALGO_LINEAR: ScaleImageCalculatorOptions.ScaleAlgorithm
    SCALE_TO_MULTIPLE_OF_FIELD_NUMBER: _ClassVar[int]
    SET_ALIGNMENT_PADDING_FIELD_NUMBER: _ClassVar[int]
    TARGET_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    TARGET_MAX_AREA_FIELD_NUMBER: _ClassVar[int]
    TARGET_WIDTH_FIELD_NUMBER: _ClassVar[int]
    USE_BT709_FIELD_NUMBER: _ClassVar[int]
    algorithm: ScaleImageCalculatorOptions.ScaleAlgorithm
    alignment_boundary: int
    ext: _descriptor.FieldDescriptor
    input_format: _image_format_pb2.ImageFormat.Format
    max_aspect_ratio: str
    min_aspect_ratio: str
    output_format: _image_format_pb2.ImageFormat.Format
    post_sharpening_coefficient: float
    preserve_aspect_ratio: bool
    scale_to_multiple_of: int
    set_alignment_padding: bool
    target_height: int
    target_max_area: int
    target_width: int
    use_bt709: bool
    def __init__(self, target_width: _Optional[int] = ..., target_height: _Optional[int] = ..., target_max_area: _Optional[int] = ..., preserve_aspect_ratio: bool = ..., min_aspect_ratio: _Optional[str] = ..., max_aspect_ratio: _Optional[str] = ..., output_format: _Optional[_Union[_image_format_pb2.ImageFormat.Format, str]] = ..., algorithm: _Optional[_Union[ScaleImageCalculatorOptions.ScaleAlgorithm, str]] = ..., alignment_boundary: _Optional[int] = ..., set_alignment_padding: bool = ..., OBSOLETE_skip_linear_rgb_conversion: bool = ..., post_sharpening_coefficient: _Optional[float] = ..., input_format: _Optional[_Union[_image_format_pb2.ImageFormat.Format, str]] = ..., scale_to_multiple_of: _Optional[int] = ..., use_bt709: bool = ...) -> None: ...
