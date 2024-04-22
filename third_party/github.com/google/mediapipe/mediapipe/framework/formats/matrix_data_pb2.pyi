from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class MatrixData(_message.Message):
    __slots__ = ["cols", "layout", "packed_data", "rows"]
    class Layout(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    COLS_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_COLUMN_MAJOR: MatrixData.Layout
    LAYOUT_FIELD_NUMBER: _ClassVar[int]
    LAYOUT_ROW_MAJOR: MatrixData.Layout
    PACKED_DATA_FIELD_NUMBER: _ClassVar[int]
    ROWS_FIELD_NUMBER: _ClassVar[int]
    cols: int
    layout: MatrixData.Layout
    packed_data: _containers.RepeatedScalarFieldContainer[float]
    rows: int
    def __init__(self, rows: _Optional[int] = ..., cols: _Optional[int] = ..., packed_data: _Optional[_Iterable[float]] = ..., layout: _Optional[_Union[MatrixData.Layout, str]] = ...) -> None: ...
