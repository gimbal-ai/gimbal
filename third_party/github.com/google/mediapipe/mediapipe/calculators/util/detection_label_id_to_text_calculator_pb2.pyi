from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util import label_map_pb2 as _label_map_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DetectionLabelIdToTextCalculatorOptions(_message.Message):
    __slots__ = ["keep_label_id", "label", "label_items", "label_map_path"]
    class LabelItemsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: int
        value: _label_map_pb2.LabelMapItem
        def __init__(self, key: _Optional[int] = ..., value: _Optional[_Union[_label_map_pb2.LabelMapItem, _Mapping]] = ...) -> None: ...
    EXT_FIELD_NUMBER: _ClassVar[int]
    KEEP_LABEL_ID_FIELD_NUMBER: _ClassVar[int]
    LABEL_FIELD_NUMBER: _ClassVar[int]
    LABEL_ITEMS_FIELD_NUMBER: _ClassVar[int]
    LABEL_MAP_PATH_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    keep_label_id: bool
    label: _containers.RepeatedScalarFieldContainer[str]
    label_items: _containers.MessageMap[int, _label_map_pb2.LabelMapItem]
    label_map_path: str
    def __init__(self, label_map_path: _Optional[str] = ..., label: _Optional[_Iterable[str]] = ..., keep_label_id: bool = ..., label_items: _Optional[_Mapping[int, _label_map_pb2.LabelMapItem]] = ...) -> None: ...
