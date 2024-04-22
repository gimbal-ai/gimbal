from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AffineModel(_message.Message):
    __slots__ = ["a", "b", "c", "d", "dx", "dy"]
    A_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    C_FIELD_NUMBER: _ClassVar[int]
    DX_FIELD_NUMBER: _ClassVar[int]
    DY_FIELD_NUMBER: _ClassVar[int]
    D_FIELD_NUMBER: _ClassVar[int]
    a: float
    b: float
    c: float
    d: float
    dx: float
    dy: float
    def __init__(self, dx: _Optional[float] = ..., dy: _Optional[float] = ..., a: _Optional[float] = ..., b: _Optional[float] = ..., c: _Optional[float] = ..., d: _Optional[float] = ...) -> None: ...

class Homography(_message.Message):
    __slots__ = ["h_00", "h_01", "h_02", "h_10", "h_11", "h_12", "h_20", "h_21"]
    H_00_FIELD_NUMBER: _ClassVar[int]
    H_01_FIELD_NUMBER: _ClassVar[int]
    H_02_FIELD_NUMBER: _ClassVar[int]
    H_10_FIELD_NUMBER: _ClassVar[int]
    H_11_FIELD_NUMBER: _ClassVar[int]
    H_12_FIELD_NUMBER: _ClassVar[int]
    H_20_FIELD_NUMBER: _ClassVar[int]
    H_21_FIELD_NUMBER: _ClassVar[int]
    h_00: float
    h_01: float
    h_02: float
    h_10: float
    h_11: float
    h_12: float
    h_20: float
    h_21: float
    def __init__(self, h_00: _Optional[float] = ..., h_01: _Optional[float] = ..., h_02: _Optional[float] = ..., h_10: _Optional[float] = ..., h_11: _Optional[float] = ..., h_12: _Optional[float] = ..., h_20: _Optional[float] = ..., h_21: _Optional[float] = ...) -> None: ...

class LinearSimilarityModel(_message.Message):
    __slots__ = ["a", "b", "dx", "dy"]
    A_FIELD_NUMBER: _ClassVar[int]
    B_FIELD_NUMBER: _ClassVar[int]
    DX_FIELD_NUMBER: _ClassVar[int]
    DY_FIELD_NUMBER: _ClassVar[int]
    a: float
    b: float
    dx: float
    dy: float
    def __init__(self, dx: _Optional[float] = ..., dy: _Optional[float] = ..., a: _Optional[float] = ..., b: _Optional[float] = ...) -> None: ...

class MixtureAffine(_message.Message):
    __slots__ = ["model"]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: _containers.RepeatedCompositeFieldContainer[AffineModel]
    def __init__(self, model: _Optional[_Iterable[_Union[AffineModel, _Mapping]]] = ...) -> None: ...

class MixtureHomography(_message.Message):
    __slots__ = ["dof", "model"]
    class VariableDOF(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    ALL_DOF: MixtureHomography.VariableDOF
    CONST_DOF: MixtureHomography.VariableDOF
    DOF_FIELD_NUMBER: _ClassVar[int]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    SKEW_ROTATION_DOF: MixtureHomography.VariableDOF
    TRANSLATION_DOF: MixtureHomography.VariableDOF
    dof: MixtureHomography.VariableDOF
    model: _containers.RepeatedCompositeFieldContainer[Homography]
    def __init__(self, model: _Optional[_Iterable[_Union[Homography, _Mapping]]] = ..., dof: _Optional[_Union[MixtureHomography.VariableDOF, str]] = ...) -> None: ...

class MixtureLinearSimilarity(_message.Message):
    __slots__ = ["model"]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: _containers.RepeatedCompositeFieldContainer[LinearSimilarityModel]
    def __init__(self, model: _Optional[_Iterable[_Union[LinearSimilarityModel, _Mapping]]] = ...) -> None: ...

class SimilarityModel(_message.Message):
    __slots__ = ["dx", "dy", "rotation", "scale"]
    DX_FIELD_NUMBER: _ClassVar[int]
    DY_FIELD_NUMBER: _ClassVar[int]
    ROTATION_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    dx: float
    dy: float
    rotation: float
    scale: float
    def __init__(self, dx: _Optional[float] = ..., dy: _Optional[float] = ..., scale: _Optional[float] = ..., rotation: _Optional[float] = ...) -> None: ...

class TranslationModel(_message.Message):
    __slots__ = ["dx", "dy"]
    DX_FIELD_NUMBER: _ClassVar[int]
    DY_FIELD_NUMBER: _ClassVar[int]
    dx: float
    dy: float
    def __init__(self, dx: _Optional[float] = ..., dy: _Optional[float] = ...) -> None: ...
