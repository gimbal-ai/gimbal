from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.util.tracking import flow_packager_pb2 as _flow_packager_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FlowPackagerCalculatorOptions(_message.Message):
    __slots__ = ["cache_file_format", "caching_chunk_size_msec", "flow_packager_options"]
    CACHE_FILE_FORMAT_FIELD_NUMBER: _ClassVar[int]
    CACHING_CHUNK_SIZE_MSEC_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    FLOW_PACKAGER_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    cache_file_format: str
    caching_chunk_size_msec: int
    ext: _descriptor.FieldDescriptor
    flow_packager_options: _flow_packager_pb2.FlowPackagerOptions
    def __init__(self, flow_packager_options: _Optional[_Union[_flow_packager_pb2.FlowPackagerOptions, _Mapping]] = ..., caching_chunk_size_msec: _Optional[int] = ..., cache_file_format: _Optional[str] = ...) -> None: ...
