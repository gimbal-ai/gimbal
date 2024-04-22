from opentelemetry.proto.logs.v1 import logs_pb2 as _logs_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ExportLogsPartialSuccess(_message.Message):
    __slots__ = ["error_message", "rejected_log_records"]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    REJECTED_LOG_RECORDS_FIELD_NUMBER: _ClassVar[int]
    error_message: str
    rejected_log_records: int
    def __init__(self, rejected_log_records: _Optional[int] = ..., error_message: _Optional[str] = ...) -> None: ...

class ExportLogsServiceRequest(_message.Message):
    __slots__ = ["resource_logs"]
    RESOURCE_LOGS_FIELD_NUMBER: _ClassVar[int]
    resource_logs: _containers.RepeatedCompositeFieldContainer[_logs_pb2.ResourceLogs]
    def __init__(self, resource_logs: _Optional[_Iterable[_Union[_logs_pb2.ResourceLogs, _Mapping]]] = ...) -> None: ...

class ExportLogsServiceResponse(_message.Message):
    __slots__ = ["partial_success"]
    PARTIAL_SUCCESS_FIELD_NUMBER: _ClassVar[int]
    partial_success: ExportLogsPartialSuccess
    def __init__(self, partial_success: _Optional[_Union[ExportLogsPartialSuccess, _Mapping]] = ...) -> None: ...
