from opentelemetry.proto.common.v1 import common_pb2 as _common_pb2
from opentelemetry.proto.resource.v1 import resource_pb2 as _resource_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor
LOG_RECORD_FLAGS_DO_NOT_USE: LogRecordFlags
LOG_RECORD_FLAGS_TRACE_FLAGS_MASK: LogRecordFlags
SEVERITY_NUMBER_DEBUG: SeverityNumber
SEVERITY_NUMBER_DEBUG2: SeverityNumber
SEVERITY_NUMBER_DEBUG3: SeverityNumber
SEVERITY_NUMBER_DEBUG4: SeverityNumber
SEVERITY_NUMBER_ERROR: SeverityNumber
SEVERITY_NUMBER_ERROR2: SeverityNumber
SEVERITY_NUMBER_ERROR3: SeverityNumber
SEVERITY_NUMBER_ERROR4: SeverityNumber
SEVERITY_NUMBER_FATAL: SeverityNumber
SEVERITY_NUMBER_FATAL2: SeverityNumber
SEVERITY_NUMBER_FATAL3: SeverityNumber
SEVERITY_NUMBER_FATAL4: SeverityNumber
SEVERITY_NUMBER_INFO: SeverityNumber
SEVERITY_NUMBER_INFO2: SeverityNumber
SEVERITY_NUMBER_INFO3: SeverityNumber
SEVERITY_NUMBER_INFO4: SeverityNumber
SEVERITY_NUMBER_TRACE: SeverityNumber
SEVERITY_NUMBER_TRACE2: SeverityNumber
SEVERITY_NUMBER_TRACE3: SeverityNumber
SEVERITY_NUMBER_TRACE4: SeverityNumber
SEVERITY_NUMBER_UNSPECIFIED: SeverityNumber
SEVERITY_NUMBER_WARN: SeverityNumber
SEVERITY_NUMBER_WARN2: SeverityNumber
SEVERITY_NUMBER_WARN3: SeverityNumber
SEVERITY_NUMBER_WARN4: SeverityNumber

class LogRecord(_message.Message):
    __slots__ = ["attributes", "body", "dropped_attributes_count", "flags", "observed_time_unix_nano", "severity_number", "severity_text", "span_id", "time_unix_nano", "trace_id"]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    BODY_FIELD_NUMBER: _ClassVar[int]
    DROPPED_ATTRIBUTES_COUNT_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    OBSERVED_TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_NUMBER_FIELD_NUMBER: _ClassVar[int]
    SEVERITY_TEXT_FIELD_NUMBER: _ClassVar[int]
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    attributes: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValue]
    body: _common_pb2.AnyValue
    dropped_attributes_count: int
    flags: int
    observed_time_unix_nano: int
    severity_number: SeverityNumber
    severity_text: str
    span_id: bytes
    time_unix_nano: int
    trace_id: bytes
    def __init__(self, time_unix_nano: _Optional[int] = ..., observed_time_unix_nano: _Optional[int] = ..., severity_number: _Optional[_Union[SeverityNumber, str]] = ..., severity_text: _Optional[str] = ..., body: _Optional[_Union[_common_pb2.AnyValue, _Mapping]] = ..., attributes: _Optional[_Iterable[_Union[_common_pb2.KeyValue, _Mapping]]] = ..., dropped_attributes_count: _Optional[int] = ..., flags: _Optional[int] = ..., trace_id: _Optional[bytes] = ..., span_id: _Optional[bytes] = ...) -> None: ...

class LogsData(_message.Message):
    __slots__ = ["resource_logs"]
    RESOURCE_LOGS_FIELD_NUMBER: _ClassVar[int]
    resource_logs: _containers.RepeatedCompositeFieldContainer[ResourceLogs]
    def __init__(self, resource_logs: _Optional[_Iterable[_Union[ResourceLogs, _Mapping]]] = ...) -> None: ...

class ResourceLogs(_message.Message):
    __slots__ = ["resource", "schema_url", "scope_logs"]
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_URL_FIELD_NUMBER: _ClassVar[int]
    SCOPE_LOGS_FIELD_NUMBER: _ClassVar[int]
    resource: _resource_pb2.Resource
    schema_url: str
    scope_logs: _containers.RepeatedCompositeFieldContainer[ScopeLogs]
    def __init__(self, resource: _Optional[_Union[_resource_pb2.Resource, _Mapping]] = ..., scope_logs: _Optional[_Iterable[_Union[ScopeLogs, _Mapping]]] = ..., schema_url: _Optional[str] = ...) -> None: ...

class ScopeLogs(_message.Message):
    __slots__ = ["log_records", "schema_url", "scope"]
    LOG_RECORDS_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_URL_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    log_records: _containers.RepeatedCompositeFieldContainer[LogRecord]
    schema_url: str
    scope: _common_pb2.InstrumentationScope
    def __init__(self, scope: _Optional[_Union[_common_pb2.InstrumentationScope, _Mapping]] = ..., log_records: _Optional[_Iterable[_Union[LogRecord, _Mapping]]] = ..., schema_url: _Optional[str] = ...) -> None: ...

class SeverityNumber(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class LogRecordFlags(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
