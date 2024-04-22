from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PacketLatencyMetricsSinkCalculatorOptions(_message.Message):
    __slots__ = ["metric_attributes", "name"]
    class MetricAttributesEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    METRIC_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    metric_attributes: _containers.ScalarMap[str, str]
    name: str
    def __init__(self, name: _Optional[str] = ..., metric_attributes: _Optional[_Mapping[str, str]] = ...) -> None: ...
