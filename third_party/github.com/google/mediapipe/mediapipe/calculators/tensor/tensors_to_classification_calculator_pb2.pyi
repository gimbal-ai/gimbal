from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import label_map_pb2 as _label_map_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class TensorsToClassificationCalculatorOptions(_message.Message):
    __slots__ = ["allow_classes", "binary_classification", "ignore_classes", "label_items", "label_map", "label_map_path", "min_score_threshold", "sort_by_descending_score", "top_k"]
    class LabelItemsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: _label_map_pb2.LabelMapItem
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[_label_map_pb2.LabelMapItem, _Mapping]] = ...) -> None: ...
    class LabelMap(_message.Message):
        __slots__ = ["entries"]
        class Entry(_message.Message):
            __slots__ = ["id", "label"]
            ID_FIELD_NUMBER: _ClassVar[int]
            LABEL_FIELD_NUMBER: _ClassVar[int]
            id: int
            label: str
            def __init__(self, id: _Optional[int] = ..., label: _Optional[str] = ...) -> None: ...
        ENTRIES_FIELD_NUMBER: _ClassVar[int]
        entries: _containers.RepeatedCompositeFieldContainer[TensorsToClassificationCalculatorOptions.LabelMap.Entry]
        def __init__(self, entries: _Optional[_Iterable[_Union[TensorsToClassificationCalculatorOptions.LabelMap.Entry, _Mapping]]] = ...) -> None: ...
    ALLOW_CLASSES_FIELD_NUMBER: _ClassVar[int]
    BINARY_CLASSIFICATION_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    IGNORE_CLASSES_FIELD_NUMBER: _ClassVar[int]
    LABEL_ITEMS_FIELD_NUMBER: _ClassVar[int]
    LABEL_MAP_FIELD_NUMBER: _ClassVar[int]
    LABEL_MAP_PATH_FIELD_NUMBER: _ClassVar[int]
    MIN_SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SORT_BY_DESCENDING_SCORE_FIELD_NUMBER: _ClassVar[int]
    TOP_K_FIELD_NUMBER: _ClassVar[int]
    allow_classes: _containers.RepeatedScalarFieldContainer[int]
    binary_classification: bool
    ext: _descriptor.FieldDescriptor
    ignore_classes: _containers.RepeatedScalarFieldContainer[int]
    label_items: _containers.MessageMap[int, _label_map_pb2.LabelMapItem]
    label_map: TensorsToClassificationCalculatorOptions.LabelMap
    label_map_path: str
    min_score_threshold: float
    sort_by_descending_score: bool
    top_k: int
    def __init__(self, min_score_threshold: _Optional[float] = ..., top_k: _Optional[int] = ..., sort_by_descending_score: bool = ..., label_map_path: _Optional[str] = ..., label_map: _Optional[_Union[TensorsToClassificationCalculatorOptions.LabelMap, _Mapping]] = ..., label_items: _Optional[_Mapping[int, _label_map_pb2.LabelMapItem]] = ..., binary_classification: bool = ..., ignore_classes: _Optional[_Iterable[int]] = ..., allow_classes: _Optional[_Iterable[int]] = ...) -> None: ...
