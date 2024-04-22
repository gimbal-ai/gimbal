from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class AffineToneModel(_message.Message):
    __slots__ = ["g_00", "g_01", "g_02", "g_03", "g_10", "g_11", "g_12", "g_13", "g_20", "g_21", "g_22", "g_23"]
    G_00_FIELD_NUMBER: _ClassVar[int]
    G_01_FIELD_NUMBER: _ClassVar[int]
    G_02_FIELD_NUMBER: _ClassVar[int]
    G_03_FIELD_NUMBER: _ClassVar[int]
    G_10_FIELD_NUMBER: _ClassVar[int]
    G_11_FIELD_NUMBER: _ClassVar[int]
    G_12_FIELD_NUMBER: _ClassVar[int]
    G_13_FIELD_NUMBER: _ClassVar[int]
    G_20_FIELD_NUMBER: _ClassVar[int]
    G_21_FIELD_NUMBER: _ClassVar[int]
    G_22_FIELD_NUMBER: _ClassVar[int]
    G_23_FIELD_NUMBER: _ClassVar[int]
    g_00: float
    g_01: float
    g_02: float
    g_03: float
    g_10: float
    g_11: float
    g_12: float
    g_13: float
    g_20: float
    g_21: float
    g_22: float
    g_23: float
    def __init__(self, g_00: _Optional[float] = ..., g_01: _Optional[float] = ..., g_02: _Optional[float] = ..., g_03: _Optional[float] = ..., g_10: _Optional[float] = ..., g_11: _Optional[float] = ..., g_12: _Optional[float] = ..., g_13: _Optional[float] = ..., g_20: _Optional[float] = ..., g_21: _Optional[float] = ..., g_22: _Optional[float] = ..., g_23: _Optional[float] = ...) -> None: ...

class GainBiasModel(_message.Message):
    __slots__ = ["bias_c1", "bias_c2", "bias_c3", "gain_c1", "gain_c2", "gain_c3"]
    BIAS_C1_FIELD_NUMBER: _ClassVar[int]
    BIAS_C2_FIELD_NUMBER: _ClassVar[int]
    BIAS_C3_FIELD_NUMBER: _ClassVar[int]
    GAIN_C1_FIELD_NUMBER: _ClassVar[int]
    GAIN_C2_FIELD_NUMBER: _ClassVar[int]
    GAIN_C3_FIELD_NUMBER: _ClassVar[int]
    bias_c1: float
    bias_c2: float
    bias_c3: float
    gain_c1: float
    gain_c2: float
    gain_c3: float
    def __init__(self, gain_c1: _Optional[float] = ..., bias_c1: _Optional[float] = ..., gain_c2: _Optional[float] = ..., bias_c2: _Optional[float] = ..., gain_c3: _Optional[float] = ..., bias_c3: _Optional[float] = ...) -> None: ...

class MixtureAffineToneModel(_message.Message):
    __slots__ = ["model"]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: _containers.RepeatedCompositeFieldContainer[AffineToneModel]
    def __init__(self, model: _Optional[_Iterable[_Union[AffineToneModel, _Mapping]]] = ...) -> None: ...

class MixtureGainBiasModel(_message.Message):
    __slots__ = ["model"]
    MODEL_FIELD_NUMBER: _ClassVar[int]
    model: _containers.RepeatedCompositeFieldContainer[GainBiasModel]
    def __init__(self, model: _Optional[_Iterable[_Union[GainBiasModel, _Mapping]]] = ...) -> None: ...
