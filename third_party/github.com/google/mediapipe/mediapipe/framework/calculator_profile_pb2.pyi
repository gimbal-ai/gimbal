from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CalculatorProfile(_message.Message):
    __slots__ = ["close_runtime", "input_stream_profiles", "name", "open_runtime", "process_input_latency", "process_output_latency", "process_runtime"]
    CLOSE_RUNTIME_FIELD_NUMBER: _ClassVar[int]
    INPUT_STREAM_PROFILES_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OPEN_RUNTIME_FIELD_NUMBER: _ClassVar[int]
    PROCESS_INPUT_LATENCY_FIELD_NUMBER: _ClassVar[int]
    PROCESS_OUTPUT_LATENCY_FIELD_NUMBER: _ClassVar[int]
    PROCESS_RUNTIME_FIELD_NUMBER: _ClassVar[int]
    close_runtime: int
    input_stream_profiles: _containers.RepeatedCompositeFieldContainer[StreamProfile]
    name: str
    open_runtime: int
    process_input_latency: TimeHistogram
    process_output_latency: TimeHistogram
    process_runtime: TimeHistogram
    def __init__(self, name: _Optional[str] = ..., open_runtime: _Optional[int] = ..., close_runtime: _Optional[int] = ..., process_runtime: _Optional[_Union[TimeHistogram, _Mapping]] = ..., process_input_latency: _Optional[_Union[TimeHistogram, _Mapping]] = ..., process_output_latency: _Optional[_Union[TimeHistogram, _Mapping]] = ..., input_stream_profiles: _Optional[_Iterable[_Union[StreamProfile, _Mapping]]] = ...) -> None: ...

class GraphProfile(_message.Message):
    __slots__ = ["calculator_profiles", "config", "graph_trace"]
    CALCULATOR_PROFILES_FIELD_NUMBER: _ClassVar[int]
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    GRAPH_TRACE_FIELD_NUMBER: _ClassVar[int]
    calculator_profiles: _containers.RepeatedCompositeFieldContainer[CalculatorProfile]
    config: _calculator_pb2.CalculatorGraphConfig
    graph_trace: _containers.RepeatedCompositeFieldContainer[GraphTrace]
    def __init__(self, graph_trace: _Optional[_Iterable[_Union[GraphTrace, _Mapping]]] = ..., calculator_profiles: _Optional[_Iterable[_Union[CalculatorProfile, _Mapping]]] = ..., config: _Optional[_Union[_calculator_pb2.CalculatorGraphConfig, _Mapping]] = ...) -> None: ...

class GraphTrace(_message.Message):
    __slots__ = ["base_time", "base_timestamp", "calculator_name", "calculator_trace", "stream_name"]
    class EventType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    class CalculatorTrace(_message.Message):
        __slots__ = ["event_type", "finish_time", "input_timestamp", "input_trace", "node_id", "output_trace", "start_time", "thread_id"]
        EVENT_TYPE_FIELD_NUMBER: _ClassVar[int]
        FINISH_TIME_FIELD_NUMBER: _ClassVar[int]
        INPUT_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
        INPUT_TRACE_FIELD_NUMBER: _ClassVar[int]
        NODE_ID_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_TRACE_FIELD_NUMBER: _ClassVar[int]
        START_TIME_FIELD_NUMBER: _ClassVar[int]
        THREAD_ID_FIELD_NUMBER: _ClassVar[int]
        event_type: GraphTrace.EventType
        finish_time: int
        input_timestamp: int
        input_trace: _containers.RepeatedCompositeFieldContainer[GraphTrace.StreamTrace]
        node_id: int
        output_trace: _containers.RepeatedCompositeFieldContainer[GraphTrace.StreamTrace]
        start_time: int
        thread_id: int
        def __init__(self, node_id: _Optional[int] = ..., input_timestamp: _Optional[int] = ..., event_type: _Optional[_Union[GraphTrace.EventType, str]] = ..., start_time: _Optional[int] = ..., finish_time: _Optional[int] = ..., input_trace: _Optional[_Iterable[_Union[GraphTrace.StreamTrace, _Mapping]]] = ..., output_trace: _Optional[_Iterable[_Union[GraphTrace.StreamTrace, _Mapping]]] = ..., thread_id: _Optional[int] = ...) -> None: ...
    class StreamTrace(_message.Message):
        __slots__ = ["event_data", "finish_time", "packet_id", "packet_timestamp", "start_time", "stream_id"]
        EVENT_DATA_FIELD_NUMBER: _ClassVar[int]
        FINISH_TIME_FIELD_NUMBER: _ClassVar[int]
        PACKET_ID_FIELD_NUMBER: _ClassVar[int]
        PACKET_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
        START_TIME_FIELD_NUMBER: _ClassVar[int]
        STREAM_ID_FIELD_NUMBER: _ClassVar[int]
        event_data: int
        finish_time: int
        packet_id: int
        packet_timestamp: int
        start_time: int
        stream_id: int
        def __init__(self, start_time: _Optional[int] = ..., finish_time: _Optional[int] = ..., packet_timestamp: _Optional[int] = ..., stream_id: _Optional[int] = ..., packet_id: _Optional[int] = ..., event_data: _Optional[int] = ...) -> None: ...
    BASE_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    BASE_TIME_FIELD_NUMBER: _ClassVar[int]
    CALCULATOR_NAME_FIELD_NUMBER: _ClassVar[int]
    CALCULATOR_TRACE_FIELD_NUMBER: _ClassVar[int]
    EVENT_TYPE_CLOSE: GraphTrace.EventType
    EVENT_TYPE_CPU_TASK_INVOKE: GraphTrace.EventType
    EVENT_TYPE_CPU_TASK_SYSTEM: GraphTrace.EventType
    EVENT_TYPE_CPU_TASK_USER: GraphTrace.EventType
    EVENT_TYPE_DSP_TASK: GraphTrace.EventType
    EVENT_TYPE_GPU_CALIBRATION: GraphTrace.EventType
    EVENT_TYPE_GPU_TASK: GraphTrace.EventType
    EVENT_TYPE_GPU_TASK_INVOKE: GraphTrace.EventType
    EVENT_TYPE_GPU_TASK_INVOKE_ADVANCED: GraphTrace.EventType
    EVENT_TYPE_NOT_READY: GraphTrace.EventType
    EVENT_TYPE_OPEN: GraphTrace.EventType
    EVENT_TYPE_PACKET_QUEUED: GraphTrace.EventType
    EVENT_TYPE_PROCESS: GraphTrace.EventType
    EVENT_TYPE_READY_FOR_CLOSE: GraphTrace.EventType
    EVENT_TYPE_READY_FOR_PROCESS: GraphTrace.EventType
    EVENT_TYPE_THROTTLED: GraphTrace.EventType
    EVENT_TYPE_TPU_TASK: GraphTrace.EventType
    EVENT_TYPE_TPU_TASK_INVOKE: GraphTrace.EventType
    EVENT_TYPE_TPU_TASK_INVOKE_ASYNC: GraphTrace.EventType
    EVENT_TYPE_UNKNOWN: GraphTrace.EventType
    EVENT_TYPE_UNTHROTTLED: GraphTrace.EventType
    STREAM_NAME_FIELD_NUMBER: _ClassVar[int]
    base_time: int
    base_timestamp: int
    calculator_name: _containers.RepeatedScalarFieldContainer[str]
    calculator_trace: _containers.RepeatedCompositeFieldContainer[GraphTrace.CalculatorTrace]
    stream_name: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, base_time: _Optional[int] = ..., base_timestamp: _Optional[int] = ..., calculator_name: _Optional[_Iterable[str]] = ..., stream_name: _Optional[_Iterable[str]] = ..., calculator_trace: _Optional[_Iterable[_Union[GraphTrace.CalculatorTrace, _Mapping]]] = ...) -> None: ...

class StreamProfile(_message.Message):
    __slots__ = ["back_edge", "latency", "name"]
    BACK_EDGE_FIELD_NUMBER: _ClassVar[int]
    LATENCY_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    back_edge: bool
    latency: TimeHistogram
    name: str
    def __init__(self, name: _Optional[str] = ..., back_edge: bool = ..., latency: _Optional[_Union[TimeHistogram, _Mapping]] = ...) -> None: ...

class TimeHistogram(_message.Message):
    __slots__ = ["count", "interval_size_usec", "num_intervals", "total"]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_SIZE_USEC_FIELD_NUMBER: _ClassVar[int]
    NUM_INTERVALS_FIELD_NUMBER: _ClassVar[int]
    TOTAL_FIELD_NUMBER: _ClassVar[int]
    count: _containers.RepeatedScalarFieldContainer[int]
    interval_size_usec: int
    num_intervals: int
    total: int
    def __init__(self, total: _Optional[int] = ..., interval_size_usec: _Optional[int] = ..., num_intervals: _Optional[int] = ..., count: _Optional[_Iterable[int]] = ...) -> None: ...
