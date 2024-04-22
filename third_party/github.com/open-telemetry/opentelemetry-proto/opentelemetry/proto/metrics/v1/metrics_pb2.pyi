from opentelemetry.proto.common.v1 import common_pb2 as _common_pb2
from opentelemetry.proto.resource.v1 import resource_pb2 as _resource_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

AGGREGATION_TEMPORALITY_CUMULATIVE: AggregationTemporality
AGGREGATION_TEMPORALITY_DELTA: AggregationTemporality
AGGREGATION_TEMPORALITY_UNSPECIFIED: AggregationTemporality
DATA_POINT_FLAGS_DO_NOT_USE: DataPointFlags
DATA_POINT_FLAGS_NO_RECORDED_VALUE_MASK: DataPointFlags
DESCRIPTOR: _descriptor.FileDescriptor

class Exemplar(_message.Message):
    __slots__ = ["as_double", "as_int", "filtered_attributes", "span_id", "time_unix_nano", "trace_id"]
    AS_DOUBLE_FIELD_NUMBER: _ClassVar[int]
    AS_INT_FIELD_NUMBER: _ClassVar[int]
    FILTERED_ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    SPAN_ID_FIELD_NUMBER: _ClassVar[int]
    TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    as_double: float
    as_int: int
    filtered_attributes: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValue]
    span_id: bytes
    time_unix_nano: int
    trace_id: bytes
    def __init__(self, filtered_attributes: _Optional[_Iterable[_Union[_common_pb2.KeyValue, _Mapping]]] = ..., time_unix_nano: _Optional[int] = ..., as_double: _Optional[float] = ..., as_int: _Optional[int] = ..., span_id: _Optional[bytes] = ..., trace_id: _Optional[bytes] = ...) -> None: ...

class ExponentialHistogram(_message.Message):
    __slots__ = ["aggregation_temporality", "data_points"]
    AGGREGATION_TEMPORALITY_FIELD_NUMBER: _ClassVar[int]
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    aggregation_temporality: AggregationTemporality
    data_points: _containers.RepeatedCompositeFieldContainer[ExponentialHistogramDataPoint]
    def __init__(self, data_points: _Optional[_Iterable[_Union[ExponentialHistogramDataPoint, _Mapping]]] = ..., aggregation_temporality: _Optional[_Union[AggregationTemporality, str]] = ...) -> None: ...

class ExponentialHistogramDataPoint(_message.Message):
    __slots__ = ["attributes", "count", "exemplars", "flags", "max", "min", "negative", "positive", "scale", "start_time_unix_nano", "sum", "time_unix_nano", "zero_count", "zero_threshold"]
    class Buckets(_message.Message):
        __slots__ = ["bucket_counts", "offset"]
        BUCKET_COUNTS_FIELD_NUMBER: _ClassVar[int]
        OFFSET_FIELD_NUMBER: _ClassVar[int]
        bucket_counts: _containers.RepeatedScalarFieldContainer[int]
        offset: int
        def __init__(self, offset: _Optional[int] = ..., bucket_counts: _Optional[_Iterable[int]] = ...) -> None: ...
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    EXEMPLARS_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    NEGATIVE_FIELD_NUMBER: _ClassVar[int]
    POSITIVE_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    START_TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    SUM_FIELD_NUMBER: _ClassVar[int]
    TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    ZERO_COUNT_FIELD_NUMBER: _ClassVar[int]
    ZERO_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    attributes: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValue]
    count: int
    exemplars: _containers.RepeatedCompositeFieldContainer[Exemplar]
    flags: int
    max: float
    min: float
    negative: ExponentialHistogramDataPoint.Buckets
    positive: ExponentialHistogramDataPoint.Buckets
    scale: int
    start_time_unix_nano: int
    sum: float
    time_unix_nano: int
    zero_count: int
    zero_threshold: float
    def __init__(self, attributes: _Optional[_Iterable[_Union[_common_pb2.KeyValue, _Mapping]]] = ..., start_time_unix_nano: _Optional[int] = ..., time_unix_nano: _Optional[int] = ..., count: _Optional[int] = ..., sum: _Optional[float] = ..., scale: _Optional[int] = ..., zero_count: _Optional[int] = ..., positive: _Optional[_Union[ExponentialHistogramDataPoint.Buckets, _Mapping]] = ..., negative: _Optional[_Union[ExponentialHistogramDataPoint.Buckets, _Mapping]] = ..., flags: _Optional[int] = ..., exemplars: _Optional[_Iterable[_Union[Exemplar, _Mapping]]] = ..., min: _Optional[float] = ..., max: _Optional[float] = ..., zero_threshold: _Optional[float] = ...) -> None: ...

class Gauge(_message.Message):
    __slots__ = ["data_points"]
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    data_points: _containers.RepeatedCompositeFieldContainer[NumberDataPoint]
    def __init__(self, data_points: _Optional[_Iterable[_Union[NumberDataPoint, _Mapping]]] = ...) -> None: ...

class Histogram(_message.Message):
    __slots__ = ["aggregation_temporality", "data_points"]
    AGGREGATION_TEMPORALITY_FIELD_NUMBER: _ClassVar[int]
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    aggregation_temporality: AggregationTemporality
    data_points: _containers.RepeatedCompositeFieldContainer[HistogramDataPoint]
    def __init__(self, data_points: _Optional[_Iterable[_Union[HistogramDataPoint, _Mapping]]] = ..., aggregation_temporality: _Optional[_Union[AggregationTemporality, str]] = ...) -> None: ...

class HistogramDataPoint(_message.Message):
    __slots__ = ["attributes", "bucket_counts", "count", "exemplars", "explicit_bounds", "flags", "max", "min", "start_time_unix_nano", "sum", "time_unix_nano"]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    BUCKET_COUNTS_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    EXEMPLARS_FIELD_NUMBER: _ClassVar[int]
    EXPLICIT_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    MIN_FIELD_NUMBER: _ClassVar[int]
    START_TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    SUM_FIELD_NUMBER: _ClassVar[int]
    TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    attributes: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValue]
    bucket_counts: _containers.RepeatedScalarFieldContainer[int]
    count: int
    exemplars: _containers.RepeatedCompositeFieldContainer[Exemplar]
    explicit_bounds: _containers.RepeatedScalarFieldContainer[float]
    flags: int
    max: float
    min: float
    start_time_unix_nano: int
    sum: float
    time_unix_nano: int
    def __init__(self, attributes: _Optional[_Iterable[_Union[_common_pb2.KeyValue, _Mapping]]] = ..., start_time_unix_nano: _Optional[int] = ..., time_unix_nano: _Optional[int] = ..., count: _Optional[int] = ..., sum: _Optional[float] = ..., bucket_counts: _Optional[_Iterable[int]] = ..., explicit_bounds: _Optional[_Iterable[float]] = ..., exemplars: _Optional[_Iterable[_Union[Exemplar, _Mapping]]] = ..., flags: _Optional[int] = ..., min: _Optional[float] = ..., max: _Optional[float] = ...) -> None: ...

class Metric(_message.Message):
    __slots__ = ["description", "exponential_histogram", "gauge", "histogram", "name", "sum", "summary", "unit"]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    EXPONENTIAL_HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    GAUGE_FIELD_NUMBER: _ClassVar[int]
    HISTOGRAM_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    SUMMARY_FIELD_NUMBER: _ClassVar[int]
    SUM_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    description: str
    exponential_histogram: ExponentialHistogram
    gauge: Gauge
    histogram: Histogram
    name: str
    sum: Sum
    summary: Summary
    unit: str
    def __init__(self, name: _Optional[str] = ..., description: _Optional[str] = ..., unit: _Optional[str] = ..., gauge: _Optional[_Union[Gauge, _Mapping]] = ..., sum: _Optional[_Union[Sum, _Mapping]] = ..., histogram: _Optional[_Union[Histogram, _Mapping]] = ..., exponential_histogram: _Optional[_Union[ExponentialHistogram, _Mapping]] = ..., summary: _Optional[_Union[Summary, _Mapping]] = ...) -> None: ...

class MetricsData(_message.Message):
    __slots__ = ["resource_metrics"]
    RESOURCE_METRICS_FIELD_NUMBER: _ClassVar[int]
    resource_metrics: _containers.RepeatedCompositeFieldContainer[ResourceMetrics]
    def __init__(self, resource_metrics: _Optional[_Iterable[_Union[ResourceMetrics, _Mapping]]] = ...) -> None: ...

class NumberDataPoint(_message.Message):
    __slots__ = ["as_double", "as_int", "attributes", "exemplars", "flags", "start_time_unix_nano", "time_unix_nano"]
    AS_DOUBLE_FIELD_NUMBER: _ClassVar[int]
    AS_INT_FIELD_NUMBER: _ClassVar[int]
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    EXEMPLARS_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    START_TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    as_double: float
    as_int: int
    attributes: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValue]
    exemplars: _containers.RepeatedCompositeFieldContainer[Exemplar]
    flags: int
    start_time_unix_nano: int
    time_unix_nano: int
    def __init__(self, attributes: _Optional[_Iterable[_Union[_common_pb2.KeyValue, _Mapping]]] = ..., start_time_unix_nano: _Optional[int] = ..., time_unix_nano: _Optional[int] = ..., as_double: _Optional[float] = ..., as_int: _Optional[int] = ..., exemplars: _Optional[_Iterable[_Union[Exemplar, _Mapping]]] = ..., flags: _Optional[int] = ...) -> None: ...

class ResourceMetrics(_message.Message):
    __slots__ = ["resource", "schema_url", "scope_metrics"]
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_URL_FIELD_NUMBER: _ClassVar[int]
    SCOPE_METRICS_FIELD_NUMBER: _ClassVar[int]
    resource: _resource_pb2.Resource
    schema_url: str
    scope_metrics: _containers.RepeatedCompositeFieldContainer[ScopeMetrics]
    def __init__(self, resource: _Optional[_Union[_resource_pb2.Resource, _Mapping]] = ..., scope_metrics: _Optional[_Iterable[_Union[ScopeMetrics, _Mapping]]] = ..., schema_url: _Optional[str] = ...) -> None: ...

class ScopeMetrics(_message.Message):
    __slots__ = ["metrics", "schema_url", "scope"]
    METRICS_FIELD_NUMBER: _ClassVar[int]
    SCHEMA_URL_FIELD_NUMBER: _ClassVar[int]
    SCOPE_FIELD_NUMBER: _ClassVar[int]
    metrics: _containers.RepeatedCompositeFieldContainer[Metric]
    schema_url: str
    scope: _common_pb2.InstrumentationScope
    def __init__(self, scope: _Optional[_Union[_common_pb2.InstrumentationScope, _Mapping]] = ..., metrics: _Optional[_Iterable[_Union[Metric, _Mapping]]] = ..., schema_url: _Optional[str] = ...) -> None: ...

class Sum(_message.Message):
    __slots__ = ["aggregation_temporality", "data_points", "is_monotonic"]
    AGGREGATION_TEMPORALITY_FIELD_NUMBER: _ClassVar[int]
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    IS_MONOTONIC_FIELD_NUMBER: _ClassVar[int]
    aggregation_temporality: AggregationTemporality
    data_points: _containers.RepeatedCompositeFieldContainer[NumberDataPoint]
    is_monotonic: bool
    def __init__(self, data_points: _Optional[_Iterable[_Union[NumberDataPoint, _Mapping]]] = ..., aggregation_temporality: _Optional[_Union[AggregationTemporality, str]] = ..., is_monotonic: bool = ...) -> None: ...

class Summary(_message.Message):
    __slots__ = ["data_points"]
    DATA_POINTS_FIELD_NUMBER: _ClassVar[int]
    data_points: _containers.RepeatedCompositeFieldContainer[SummaryDataPoint]
    def __init__(self, data_points: _Optional[_Iterable[_Union[SummaryDataPoint, _Mapping]]] = ...) -> None: ...

class SummaryDataPoint(_message.Message):
    __slots__ = ["attributes", "count", "flags", "quantile_values", "start_time_unix_nano", "sum", "time_unix_nano"]
    class ValueAtQuantile(_message.Message):
        __slots__ = ["quantile", "value"]
        QUANTILE_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        quantile: float
        value: float
        def __init__(self, quantile: _Optional[float] = ..., value: _Optional[float] = ...) -> None: ...
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    FLAGS_FIELD_NUMBER: _ClassVar[int]
    QUANTILE_VALUES_FIELD_NUMBER: _ClassVar[int]
    START_TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    SUM_FIELD_NUMBER: _ClassVar[int]
    TIME_UNIX_NANO_FIELD_NUMBER: _ClassVar[int]
    attributes: _containers.RepeatedCompositeFieldContainer[_common_pb2.KeyValue]
    count: int
    flags: int
    quantile_values: _containers.RepeatedCompositeFieldContainer[SummaryDataPoint.ValueAtQuantile]
    start_time_unix_nano: int
    sum: float
    time_unix_nano: int
    def __init__(self, attributes: _Optional[_Iterable[_Union[_common_pb2.KeyValue, _Mapping]]] = ..., start_time_unix_nano: _Optional[int] = ..., time_unix_nano: _Optional[int] = ..., count: _Optional[int] = ..., sum: _Optional[float] = ..., quantile_values: _Optional[_Iterable[_Union[SummaryDataPoint.ValueAtQuantile, _Mapping]]] = ..., flags: _Optional[int] = ...) -> None: ...

class AggregationTemporality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class DataPointFlags(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
