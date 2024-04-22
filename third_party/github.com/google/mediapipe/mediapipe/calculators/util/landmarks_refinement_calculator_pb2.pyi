from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class LandmarksRefinementCalculatorOptions(_message.Message):
    __slots__ = ["refinement"]
    class Refinement(_message.Message):
        __slots__ = ["indexes_mapping", "z_refinement"]
        INDEXES_MAPPING_FIELD_NUMBER: _ClassVar[int]
        Z_REFINEMENT_FIELD_NUMBER: _ClassVar[int]
        indexes_mapping: _containers.RepeatedScalarFieldContainer[int]
        z_refinement: LandmarksRefinementCalculatorOptions.ZRefinement
        def __init__(self, indexes_mapping: _Optional[_Iterable[int]] = ..., z_refinement: _Optional[_Union[LandmarksRefinementCalculatorOptions.ZRefinement, _Mapping]] = ...) -> None: ...
    class ZRefinement(_message.Message):
        __slots__ = ["assign_average", "copy", "none"]
        ASSIGN_AVERAGE_FIELD_NUMBER: _ClassVar[int]
        COPY_FIELD_NUMBER: _ClassVar[int]
        NONE_FIELD_NUMBER: _ClassVar[int]
        assign_average: LandmarksRefinementCalculatorOptions.ZRefinementAssignAverage
        copy: LandmarksRefinementCalculatorOptions.ZRefinementCopy
        none: LandmarksRefinementCalculatorOptions.ZRefinementNone
        def __init__(self, none: _Optional[_Union[LandmarksRefinementCalculatorOptions.ZRefinementNone, _Mapping]] = ..., copy: _Optional[_Union[LandmarksRefinementCalculatorOptions.ZRefinementCopy, _Mapping]] = ..., assign_average: _Optional[_Union[LandmarksRefinementCalculatorOptions.ZRefinementAssignAverage, _Mapping]] = ...) -> None: ...
    class ZRefinementAssignAverage(_message.Message):
        __slots__ = ["indexes_for_average"]
        INDEXES_FOR_AVERAGE_FIELD_NUMBER: _ClassVar[int]
        indexes_for_average: _containers.RepeatedScalarFieldContainer[int]
        def __init__(self, indexes_for_average: _Optional[_Iterable[int]] = ...) -> None: ...
    class ZRefinementCopy(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    class ZRefinementNone(_message.Message):
        __slots__ = []
        def __init__(self) -> None: ...
    EXT_FIELD_NUMBER: _ClassVar[int]
    REFINEMENT_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    refinement: _containers.RepeatedCompositeFieldContainer[LandmarksRefinementCalculatorOptions.Refinement]
    def __init__(self, refinement: _Optional[_Iterable[_Union[LandmarksRefinementCalculatorOptions.Refinement, _Mapping]]] = ...) -> None: ...
