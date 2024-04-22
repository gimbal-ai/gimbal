from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import any_pb2 as _any_pb2
from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from mediapipe.framework import packet_factory_pb2 as _packet_factory_pb2
from mediapipe.framework import packet_generator_pb2 as _packet_generator_pb2
from mediapipe.framework import status_handler_pb2 as _status_handler_pb2
from mediapipe.framework import stream_handler_pb2 as _stream_handler_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

from mediapipe.framework.calculator_options_pb2 import CalculatorOptions
DESCRIPTOR: _descriptor.FileDescriptor

class CalculatorGraphConfig(_message.Message):
    __slots__ = ["executor", "graph_options", "input_side_packet", "input_stream", "input_stream_handler", "max_queue_size", "node", "num_threads", "options", "output_side_packet", "output_stream", "output_stream_handler", "package", "packet_factory", "packet_generator", "profiler_config", "report_deadlock", "status_handler", "type"]
    class Node(_message.Message):
        __slots__ = ["buffer_size_hint", "calculator", "executor", "external_input", "input_side_packet", "input_stream", "input_stream_handler", "input_stream_info", "max_in_flight", "name", "node_options", "option_value", "options", "output_side_packet", "output_stream", "output_stream_handler", "profiler_config", "source_layer"]
        BUFFER_SIZE_HINT_FIELD_NUMBER: _ClassVar[int]
        CALCULATOR_FIELD_NUMBER: _ClassVar[int]
        EXECUTOR_FIELD_NUMBER: _ClassVar[int]
        EXTERNAL_INPUT_FIELD_NUMBER: _ClassVar[int]
        INPUT_SIDE_PACKET_FIELD_NUMBER: _ClassVar[int]
        INPUT_STREAM_FIELD_NUMBER: _ClassVar[int]
        INPUT_STREAM_HANDLER_FIELD_NUMBER: _ClassVar[int]
        INPUT_STREAM_INFO_FIELD_NUMBER: _ClassVar[int]
        MAX_IN_FLIGHT_FIELD_NUMBER: _ClassVar[int]
        NAME_FIELD_NUMBER: _ClassVar[int]
        NODE_OPTIONS_FIELD_NUMBER: _ClassVar[int]
        OPTIONS_FIELD_NUMBER: _ClassVar[int]
        OPTION_VALUE_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_SIDE_PACKET_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_STREAM_FIELD_NUMBER: _ClassVar[int]
        OUTPUT_STREAM_HANDLER_FIELD_NUMBER: _ClassVar[int]
        PROFILER_CONFIG_FIELD_NUMBER: _ClassVar[int]
        SOURCE_LAYER_FIELD_NUMBER: _ClassVar[int]
        buffer_size_hint: int
        calculator: str
        executor: str
        external_input: _containers.RepeatedScalarFieldContainer[str]
        input_side_packet: _containers.RepeatedScalarFieldContainer[str]
        input_stream: _containers.RepeatedScalarFieldContainer[str]
        input_stream_handler: _stream_handler_pb2.InputStreamHandlerConfig
        input_stream_info: _containers.RepeatedCompositeFieldContainer[InputStreamInfo]
        max_in_flight: int
        name: str
        node_options: _containers.RepeatedCompositeFieldContainer[_any_pb2.Any]
        option_value: _containers.RepeatedScalarFieldContainer[str]
        options: _calculator_options_pb2.CalculatorOptions
        output_side_packet: _containers.RepeatedScalarFieldContainer[str]
        output_stream: _containers.RepeatedScalarFieldContainer[str]
        output_stream_handler: _stream_handler_pb2.OutputStreamHandlerConfig
        profiler_config: ProfilerConfig
        source_layer: int
        def __init__(self, name: _Optional[str] = ..., calculator: _Optional[str] = ..., input_stream: _Optional[_Iterable[str]] = ..., output_stream: _Optional[_Iterable[str]] = ..., input_side_packet: _Optional[_Iterable[str]] = ..., output_side_packet: _Optional[_Iterable[str]] = ..., options: _Optional[_Union[_calculator_options_pb2.CalculatorOptions, _Mapping]] = ..., node_options: _Optional[_Iterable[_Union[_any_pb2.Any, _Mapping]]] = ..., source_layer: _Optional[int] = ..., buffer_size_hint: _Optional[int] = ..., input_stream_handler: _Optional[_Union[_stream_handler_pb2.InputStreamHandlerConfig, _Mapping]] = ..., output_stream_handler: _Optional[_Union[_stream_handler_pb2.OutputStreamHandlerConfig, _Mapping]] = ..., input_stream_info: _Optional[_Iterable[_Union[InputStreamInfo, _Mapping]]] = ..., executor: _Optional[str] = ..., profiler_config: _Optional[_Union[ProfilerConfig, _Mapping]] = ..., max_in_flight: _Optional[int] = ..., option_value: _Optional[_Iterable[str]] = ..., external_input: _Optional[_Iterable[str]] = ...) -> None: ...
    EXECUTOR_FIELD_NUMBER: _ClassVar[int]
    GRAPH_OPTIONS_FIELD_NUMBER: _ClassVar[int]
    INPUT_SIDE_PACKET_FIELD_NUMBER: _ClassVar[int]
    INPUT_STREAM_FIELD_NUMBER: _ClassVar[int]
    INPUT_STREAM_HANDLER_FIELD_NUMBER: _ClassVar[int]
    MAX_QUEUE_SIZE_FIELD_NUMBER: _ClassVar[int]
    NODE_FIELD_NUMBER: _ClassVar[int]
    NUM_THREADS_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_SIDE_PACKET_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_STREAM_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_STREAM_HANDLER_FIELD_NUMBER: _ClassVar[int]
    PACKAGE_FIELD_NUMBER: _ClassVar[int]
    PACKET_FACTORY_FIELD_NUMBER: _ClassVar[int]
    PACKET_GENERATOR_FIELD_NUMBER: _ClassVar[int]
    PROFILER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    REPORT_DEADLOCK_FIELD_NUMBER: _ClassVar[int]
    STATUS_HANDLER_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    executor: _containers.RepeatedCompositeFieldContainer[ExecutorConfig]
    graph_options: _containers.RepeatedCompositeFieldContainer[_any_pb2.Any]
    input_side_packet: _containers.RepeatedScalarFieldContainer[str]
    input_stream: _containers.RepeatedScalarFieldContainer[str]
    input_stream_handler: _stream_handler_pb2.InputStreamHandlerConfig
    max_queue_size: int
    node: _containers.RepeatedCompositeFieldContainer[CalculatorGraphConfig.Node]
    num_threads: int
    options: _mediapipe_options_pb2.MediaPipeOptions
    output_side_packet: _containers.RepeatedScalarFieldContainer[str]
    output_stream: _containers.RepeatedScalarFieldContainer[str]
    output_stream_handler: _stream_handler_pb2.OutputStreamHandlerConfig
    package: str
    packet_factory: _containers.RepeatedCompositeFieldContainer[_packet_factory_pb2.PacketFactoryConfig]
    packet_generator: _containers.RepeatedCompositeFieldContainer[_packet_generator_pb2.PacketGeneratorConfig]
    profiler_config: ProfilerConfig
    report_deadlock: bool
    status_handler: _containers.RepeatedCompositeFieldContainer[_status_handler_pb2.StatusHandlerConfig]
    type: str
    def __init__(self, node: _Optional[_Iterable[_Union[CalculatorGraphConfig.Node, _Mapping]]] = ..., packet_factory: _Optional[_Iterable[_Union[_packet_factory_pb2.PacketFactoryConfig, _Mapping]]] = ..., packet_generator: _Optional[_Iterable[_Union[_packet_generator_pb2.PacketGeneratorConfig, _Mapping]]] = ..., num_threads: _Optional[int] = ..., status_handler: _Optional[_Iterable[_Union[_status_handler_pb2.StatusHandlerConfig, _Mapping]]] = ..., input_stream: _Optional[_Iterable[str]] = ..., output_stream: _Optional[_Iterable[str]] = ..., input_side_packet: _Optional[_Iterable[str]] = ..., output_side_packet: _Optional[_Iterable[str]] = ..., max_queue_size: _Optional[int] = ..., report_deadlock: bool = ..., input_stream_handler: _Optional[_Union[_stream_handler_pb2.InputStreamHandlerConfig, _Mapping]] = ..., output_stream_handler: _Optional[_Union[_stream_handler_pb2.OutputStreamHandlerConfig, _Mapping]] = ..., executor: _Optional[_Iterable[_Union[ExecutorConfig, _Mapping]]] = ..., profiler_config: _Optional[_Union[ProfilerConfig, _Mapping]] = ..., package: _Optional[str] = ..., type: _Optional[str] = ..., options: _Optional[_Union[_mediapipe_options_pb2.MediaPipeOptions, _Mapping]] = ..., graph_options: _Optional[_Iterable[_Union[_any_pb2.Any, _Mapping]]] = ...) -> None: ...

class ExecutorConfig(_message.Message):
    __slots__ = ["name", "options", "type"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    OPTIONS_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    name: str
    options: _mediapipe_options_pb2.MediaPipeOptions
    type: str
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., options: _Optional[_Union[_mediapipe_options_pb2.MediaPipeOptions, _Mapping]] = ...) -> None: ...

class InputCollection(_message.Message):
    __slots__ = ["external_input_name", "file_name", "input_type", "name", "side_packet_name"]
    class InputType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    EXTERNAL_INPUT_NAME_FIELD_NUMBER: _ClassVar[int]
    FILE_NAME_FIELD_NUMBER: _ClassVar[int]
    INPUT_TYPE_FIELD_NUMBER: _ClassVar[int]
    INPUT_TYPE_FOREIGN_CSV_TEXT: InputCollection.InputType
    INPUT_TYPE_FOREIGN_RECORDIO: InputCollection.InputType
    INPUT_TYPE_INVALID_UPPER_BOUND: InputCollection.InputType
    INPUT_TYPE_RECORDIO: InputCollection.InputType
    INPUT_TYPE_UNKNOWN: InputCollection.InputType
    NAME_FIELD_NUMBER: _ClassVar[int]
    SIDE_PACKET_NAME_FIELD_NUMBER: _ClassVar[int]
    external_input_name: _containers.RepeatedScalarFieldContainer[str]
    file_name: str
    input_type: InputCollection.InputType
    name: str
    side_packet_name: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, name: _Optional[str] = ..., side_packet_name: _Optional[_Iterable[str]] = ..., external_input_name: _Optional[_Iterable[str]] = ..., input_type: _Optional[_Union[InputCollection.InputType, str]] = ..., file_name: _Optional[str] = ...) -> None: ...

class InputCollectionSet(_message.Message):
    __slots__ = ["input_collection"]
    INPUT_COLLECTION_FIELD_NUMBER: _ClassVar[int]
    input_collection: _containers.RepeatedCompositeFieldContainer[InputCollection]
    def __init__(self, input_collection: _Optional[_Iterable[_Union[InputCollection, _Mapping]]] = ...) -> None: ...

class InputStreamInfo(_message.Message):
    __slots__ = ["back_edge", "tag_index"]
    BACK_EDGE_FIELD_NUMBER: _ClassVar[int]
    TAG_INDEX_FIELD_NUMBER: _ClassVar[int]
    back_edge: bool
    tag_index: str
    def __init__(self, tag_index: _Optional[str] = ..., back_edge: bool = ...) -> None: ...

class ProfilerConfig(_message.Message):
    __slots__ = ["calculator_filter", "enable_input_output_latency", "enable_profiler", "enable_stream_latency", "histogram_interval_size_usec", "num_histogram_intervals", "trace_enabled", "trace_event_types_disabled", "trace_log_capacity", "trace_log_count", "trace_log_disabled", "trace_log_duration_events", "trace_log_instant_events", "trace_log_interval_count", "trace_log_interval_usec", "trace_log_margin_usec", "trace_log_path", "use_packet_timestamp_for_added_packet"]
    CALCULATOR_FILTER_FIELD_NUMBER: _ClassVar[int]
    ENABLE_INPUT_OUTPUT_LATENCY_FIELD_NUMBER: _ClassVar[int]
    ENABLE_PROFILER_FIELD_NUMBER: _ClassVar[int]
    ENABLE_STREAM_LATENCY_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAM_INTERVAL_SIZE_USEC_FIELD_NUMBER: _ClassVar[int]
    NUM_HISTOGRAM_INTERVALS_FIELD_NUMBER: _ClassVar[int]
    TRACE_ENABLED_FIELD_NUMBER: _ClassVar[int]
    TRACE_EVENT_TYPES_DISABLED_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOG_CAPACITY_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOG_COUNT_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOG_DISABLED_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOG_DURATION_EVENTS_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOG_INSTANT_EVENTS_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOG_INTERVAL_COUNT_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOG_INTERVAL_USEC_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOG_MARGIN_USEC_FIELD_NUMBER: _ClassVar[int]
    TRACE_LOG_PATH_FIELD_NUMBER: _ClassVar[int]
    USE_PACKET_TIMESTAMP_FOR_ADDED_PACKET_FIELD_NUMBER: _ClassVar[int]
    calculator_filter: str
    enable_input_output_latency: bool
    enable_profiler: bool
    enable_stream_latency: bool
    histogram_interval_size_usec: int
    num_histogram_intervals: int
    trace_enabled: bool
    trace_event_types_disabled: _containers.RepeatedScalarFieldContainer[int]
    trace_log_capacity: int
    trace_log_count: int
    trace_log_disabled: bool
    trace_log_duration_events: bool
    trace_log_instant_events: bool
    trace_log_interval_count: int
    trace_log_interval_usec: int
    trace_log_margin_usec: int
    trace_log_path: str
    use_packet_timestamp_for_added_packet: bool
    def __init__(self, histogram_interval_size_usec: _Optional[int] = ..., num_histogram_intervals: _Optional[int] = ..., enable_input_output_latency: bool = ..., enable_profiler: bool = ..., enable_stream_latency: bool = ..., use_packet_timestamp_for_added_packet: bool = ..., trace_log_capacity: _Optional[int] = ..., trace_event_types_disabled: _Optional[_Iterable[int]] = ..., trace_log_path: _Optional[str] = ..., trace_log_count: _Optional[int] = ..., trace_log_interval_usec: _Optional[int] = ..., trace_log_margin_usec: _Optional[int] = ..., trace_log_duration_events: bool = ..., trace_log_interval_count: _Optional[int] = ..., trace_log_disabled: bool = ..., trace_enabled: bool = ..., trace_log_instant_events: bool = ..., calculator_filter: _Optional[str] = ...) -> None: ...
