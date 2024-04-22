from mediapipe.util.tracking import motion_models_pb2 as _motion_models_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CameraMotion(_message.Message):
    __slots__ = ["affine", "average_homography_error", "average_magnitude", "blur_score", "bluriness", "flags", "frac_long_features_rejected", "frame_height", "frame_width", "homography", "homography_inlier_coverage", "homography_strict_inlier_coverage", "linear_similarity", "match_frame", "mixture_homography", "mixture_homography_spectrum", "mixture_inlier_coverage", "mixture_row_sigma", "overlay_domain", "overlay_indices", "overridden_type", "rolling_shutter_guess", "rolling_shutter_motion_index", "similarity", "similarity_inlier_ratio", "similarity_strict_inlier_ratio", "timestamp_usec", "translation", "translation_variance", "type"]
    class Flags(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class Type(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    AFFINE_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_HOMOGRAPHY_ERROR_FIELD_NUMBER: _ClassVar[int]
    AVERAGE_MAGNITUDE_FIELD_NUMBER: _ClassVar[int]
    BLURINESS_FIELD_NUMBER: _ClassVar[int]
    BLUR_SCORE_FIELD_NUMBER: _ClassVar[int]
    CAMERA_MOTION_FLAG_BLURRY_FRAME: CameraMotion.Flags
    CAMERA_MOTION_FLAG_CENTER_FRAME: CameraMotion.Flags
    CAMERA_MOTION_FLAG_DUPLICATED: CameraMotion.Flags
    CAMERA_MOTION_FLAG_MAJOR_OVERLAY: CameraMotion.Flags
    CAMERA_MOTION_FLAG_SHARP_FRAME: CameraMotion.Flags
    CAMERA_MOTION_FLAG_SHOT_BOUNDARY: CameraMotion.Flags
    CAMERA_MOTION_FLAG_SHOT_FADE: CameraMotion.Flags
    CAMERA_MOTION_FLAG_SINGULAR_ESTIMATION: CameraMotion.Flags
    CAMERA_MOTION_FLAG_UNKNOWN: CameraMotion.Flags
    CAMERA_MOTION_TYPE_INVALID: CameraMotion.Type
    CAMERA_MOTION_TYPE_UNSTABLE: CameraMotion.Type
    CAMERA_MOTION_TYPE_UNSTABLE_HOMOG: CameraMotion.Type
    CAMERA_MOTION_TYPE_UNSTABLE_SIM: CameraMotion.Type
    CAMERA_MOTION_TYPE_VALID: CameraMotion.Type
    Extensions: _python_message._ExtensionDict
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    FRAC_LONG_FEATURES_REJECTED_FIELD_NUMBER: _ClassVar[int]
    FRAME_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FRAME_WIDTH_FIELD_NUMBER: _ClassVar[int]
    HOMOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    HOMOGRAPHY_INLIER_COVERAGE_FIELD_NUMBER: _ClassVar[int]
    HOMOGRAPHY_STRICT_INLIER_COVERAGE_FIELD_NUMBER: _ClassVar[int]
    LINEAR_SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    MATCH_FRAME_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_HOMOGRAPHY_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_HOMOGRAPHY_SPECTRUM_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_INLIER_COVERAGE_FIELD_NUMBER: _ClassVar[int]
    MIXTURE_ROW_SIGMA_FIELD_NUMBER: _ClassVar[int]
    OVERLAY_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    OVERLAY_INDICES_FIELD_NUMBER: _ClassVar[int]
    OVERRIDDEN_TYPE_FIELD_NUMBER: _ClassVar[int]
    ROLLING_SHUTTER_GUESS_FIELD_NUMBER: _ClassVar[int]
    ROLLING_SHUTTER_MOTION_INDEX_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_INLIER_RATIO_FIELD_NUMBER: _ClassVar[int]
    SIMILARITY_STRICT_INLIER_RATIO_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_USEC_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_FIELD_NUMBER: _ClassVar[int]
    TRANSLATION_VARIANCE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    affine: _motion_models_pb2.AffineModel
    average_homography_error: float
    average_magnitude: float
    blur_score: float
    bluriness: float
    flags: int
    frac_long_features_rejected: float
    frame_height: float
    frame_width: float
    homography: _motion_models_pb2.Homography
    homography_inlier_coverage: float
    homography_strict_inlier_coverage: float
    linear_similarity: _motion_models_pb2.LinearSimilarityModel
    match_frame: int
    mixture_homography: _motion_models_pb2.MixtureHomography
    mixture_homography_spectrum: _containers.RepeatedCompositeFieldContainer[_motion_models_pb2.MixtureHomography]
    mixture_inlier_coverage: _containers.RepeatedScalarFieldContainer[float]
    mixture_row_sigma: float
    overlay_domain: int
    overlay_indices: _containers.RepeatedScalarFieldContainer[int]
    overridden_type: CameraMotion.Type
    rolling_shutter_guess: float
    rolling_shutter_motion_index: int
    similarity: _motion_models_pb2.SimilarityModel
    similarity_inlier_ratio: float
    similarity_strict_inlier_ratio: float
    timestamp_usec: int
    translation: _motion_models_pb2.TranslationModel
    translation_variance: float
    type: CameraMotion.Type
    def __init__(self, translation: _Optional[_Union[_motion_models_pb2.TranslationModel, _Mapping]] = ..., similarity: _Optional[_Union[_motion_models_pb2.SimilarityModel, _Mapping]] = ..., linear_similarity: _Optional[_Union[_motion_models_pb2.LinearSimilarityModel, _Mapping]] = ..., affine: _Optional[_Union[_motion_models_pb2.AffineModel, _Mapping]] = ..., homography: _Optional[_Union[_motion_models_pb2.Homography, _Mapping]] = ..., mixture_homography: _Optional[_Union[_motion_models_pb2.MixtureHomography, _Mapping]] = ..., frame_width: _Optional[float] = ..., frame_height: _Optional[float] = ..., mixture_homography_spectrum: _Optional[_Iterable[_Union[_motion_models_pb2.MixtureHomography, _Mapping]]] = ..., mixture_row_sigma: _Optional[float] = ..., average_magnitude: _Optional[float] = ..., translation_variance: _Optional[float] = ..., similarity_inlier_ratio: _Optional[float] = ..., similarity_strict_inlier_ratio: _Optional[float] = ..., average_homography_error: _Optional[float] = ..., homography_inlier_coverage: _Optional[float] = ..., homography_strict_inlier_coverage: _Optional[float] = ..., mixture_inlier_coverage: _Optional[_Iterable[float]] = ..., rolling_shutter_guess: _Optional[float] = ..., rolling_shutter_motion_index: _Optional[int] = ..., overlay_indices: _Optional[_Iterable[int]] = ..., overlay_domain: _Optional[int] = ..., type: _Optional[_Union[CameraMotion.Type, str]] = ..., overridden_type: _Optional[_Union[CameraMotion.Type, str]] = ..., flags: _Optional[int] = ..., blur_score: _Optional[float] = ..., bluriness: _Optional[float] = ..., frac_long_features_rejected: _Optional[float] = ..., timestamp_usec: _Optional[int] = ..., match_frame: _Optional[int] = ...) -> None: ...
