from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.framework.formats import classification_pb2 as _classification_pb2
from mediapipe.framework.formats import landmark_pb2 as _landmark_pb2
from mediapipe.framework.formats import matrix_data_pb2 as _matrix_data_pb2
from mediapipe.framework.formats import time_series_header_pb2 as _time_series_header_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ConstantSidePacketCalculatorOptions(_message.Message):
    __slots__ = ["packet"]
    class ConstantSidePacket(_message.Message):
        __slots__ = ["bool_value", "classification_list_value", "double_value", "float_value", "int64_value", "int_value", "landmark_list_value", "matrix_data_value", "string_value", "time_series_header_value", "uint64_value"]
        BOOL_VALUE_FIELD_NUMBER: _ClassVar[int]
        CLASSIFICATION_LIST_VALUE_FIELD_NUMBER: _ClassVar[int]
        DOUBLE_VALUE_FIELD_NUMBER: _ClassVar[int]
        FLOAT_VALUE_FIELD_NUMBER: _ClassVar[int]
        INT64_VALUE_FIELD_NUMBER: _ClassVar[int]
        INT_VALUE_FIELD_NUMBER: _ClassVar[int]
        LANDMARK_LIST_VALUE_FIELD_NUMBER: _ClassVar[int]
        MATRIX_DATA_VALUE_FIELD_NUMBER: _ClassVar[int]
        STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
        TIME_SERIES_HEADER_VALUE_FIELD_NUMBER: _ClassVar[int]
        UINT64_VALUE_FIELD_NUMBER: _ClassVar[int]
        bool_value: bool
        classification_list_value: _classification_pb2.ClassificationList
        double_value: float
        float_value: float
        int64_value: int
        int_value: int
        landmark_list_value: _landmark_pb2.LandmarkList
        matrix_data_value: _matrix_data_pb2.MatrixData
        string_value: str
        time_series_header_value: _time_series_header_pb2.TimeSeriesHeader
        uint64_value: int
        def __init__(self, int_value: _Optional[int] = ..., uint64_value: _Optional[int] = ..., int64_value: _Optional[int] = ..., float_value: _Optional[float] = ..., double_value: _Optional[float] = ..., bool_value: bool = ..., string_value: _Optional[str] = ..., classification_list_value: _Optional[_Union[_classification_pb2.ClassificationList, _Mapping]] = ..., landmark_list_value: _Optional[_Union[_landmark_pb2.LandmarkList, _Mapping]] = ..., time_series_header_value: _Optional[_Union[_time_series_header_pb2.TimeSeriesHeader, _Mapping]] = ..., matrix_data_value: _Optional[_Union[_matrix_data_pb2.MatrixData, _Mapping]] = ...) -> None: ...
    PACKET_FIELD_NUMBER: _ClassVar[int]
    packet: _containers.RepeatedCompositeFieldContainer[ConstantSidePacketCalculatorOptions.ConstantSidePacket]
    def __init__(self, packet: _Optional[_Iterable[_Union[ConstantSidePacketCalculatorOptions.ConstantSidePacket, _Mapping]]] = ...) -> None: ...
