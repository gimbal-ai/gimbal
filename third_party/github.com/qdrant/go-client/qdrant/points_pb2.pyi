from qdrant import collections_pb2 as _collections_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from qdrant import json_with_int_pb2 as _json_with_int_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

Acknowledged: UpdateStatus
All: ReadConsistencyType
Asc: Direction
AverageVector: RecommendStrategy
BestScore: RecommendStrategy
ClockRejected: UpdateStatus
Completed: UpdateStatus
DESCRIPTOR: _descriptor.FileDescriptor
Desc: Direction
FieldTypeBool: FieldType
FieldTypeDatetime: FieldType
FieldTypeFloat: FieldType
FieldTypeGeo: FieldType
FieldTypeInteger: FieldType
FieldTypeKeyword: FieldType
FieldTypeText: FieldType
Majority: ReadConsistencyType
Medium: WriteOrderingType
Quorum: ReadConsistencyType
RRF: Fusion
Strong: WriteOrderingType
UnknownUpdateStatus: UpdateStatus
Weak: WriteOrderingType

class BatchResult(_message.Message):
    __slots__ = ["result"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[ScoredPoint]
    def __init__(self, result: _Optional[_Iterable[_Union[ScoredPoint, _Mapping]]] = ...) -> None: ...

class ClearPayloadPoints(_message.Message):
    __slots__ = ["collection_name", "ordering", "points", "shard_key_selector", "wait"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    ordering: WriteOrdering
    points: PointsSelector
    shard_key_selector: ShardKeySelector
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., points: _Optional[_Union[PointsSelector, _Mapping]] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class Condition(_message.Message):
    __slots__ = ["field", "filter", "has_id", "is_empty", "is_null", "nested"]
    FIELD_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    HAS_ID_FIELD_NUMBER: _ClassVar[int]
    IS_EMPTY_FIELD_NUMBER: _ClassVar[int]
    IS_NULL_FIELD_NUMBER: _ClassVar[int]
    NESTED_FIELD_NUMBER: _ClassVar[int]
    field: FieldCondition
    filter: Filter
    has_id: HasIdCondition
    is_empty: IsEmptyCondition
    is_null: IsNullCondition
    nested: NestedCondition
    def __init__(self, field: _Optional[_Union[FieldCondition, _Mapping]] = ..., is_empty: _Optional[_Union[IsEmptyCondition, _Mapping]] = ..., has_id: _Optional[_Union[HasIdCondition, _Mapping]] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., is_null: _Optional[_Union[IsNullCondition, _Mapping]] = ..., nested: _Optional[_Union[NestedCondition, _Mapping]] = ...) -> None: ...

class ContextExamplePair(_message.Message):
    __slots__ = ["negative", "positive"]
    NEGATIVE_FIELD_NUMBER: _ClassVar[int]
    POSITIVE_FIELD_NUMBER: _ClassVar[int]
    negative: VectorExample
    positive: VectorExample
    def __init__(self, positive: _Optional[_Union[VectorExample, _Mapping]] = ..., negative: _Optional[_Union[VectorExample, _Mapping]] = ...) -> None: ...

class ContextInput(_message.Message):
    __slots__ = ["pairs"]
    PAIRS_FIELD_NUMBER: _ClassVar[int]
    pairs: _containers.RepeatedCompositeFieldContainer[ContextInputPair]
    def __init__(self, pairs: _Optional[_Iterable[_Union[ContextInputPair, _Mapping]]] = ...) -> None: ...

class ContextInputPair(_message.Message):
    __slots__ = ["negative", "positive"]
    NEGATIVE_FIELD_NUMBER: _ClassVar[int]
    POSITIVE_FIELD_NUMBER: _ClassVar[int]
    negative: VectorInput
    positive: VectorInput
    def __init__(self, positive: _Optional[_Union[VectorInput, _Mapping]] = ..., negative: _Optional[_Union[VectorInput, _Mapping]] = ...) -> None: ...

class CountPoints(_message.Message):
    __slots__ = ["collection_name", "exact", "filter", "read_consistency", "shard_key_selector"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    EXACT_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    exact: bool
    filter: Filter
    read_consistency: ReadConsistency
    shard_key_selector: ShardKeySelector
    def __init__(self, collection_name: _Optional[str] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., exact: bool = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class CountResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: CountResult
    time: float
    def __init__(self, result: _Optional[_Union[CountResult, _Mapping]] = ..., time: _Optional[float] = ...) -> None: ...

class CountResult(_message.Message):
    __slots__ = ["count"]
    COUNT_FIELD_NUMBER: _ClassVar[int]
    count: int
    def __init__(self, count: _Optional[int] = ...) -> None: ...

class CreateFieldIndexCollection(_message.Message):
    __slots__ = ["collection_name", "field_index_params", "field_name", "field_type", "ordering", "wait"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELD_INDEX_PARAMS_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELD_TYPE_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    field_index_params: _collections_pb2.PayloadIndexParams
    field_name: str
    field_type: FieldType
    ordering: WriteOrdering
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., field_name: _Optional[str] = ..., field_type: _Optional[_Union[FieldType, str]] = ..., field_index_params: _Optional[_Union[_collections_pb2.PayloadIndexParams, _Mapping]] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ...) -> None: ...

class DatetimeRange(_message.Message):
    __slots__ = ["gt", "gte", "lt", "lte"]
    GTE_FIELD_NUMBER: _ClassVar[int]
    GT_FIELD_NUMBER: _ClassVar[int]
    LTE_FIELD_NUMBER: _ClassVar[int]
    LT_FIELD_NUMBER: _ClassVar[int]
    gt: _timestamp_pb2.Timestamp
    gte: _timestamp_pb2.Timestamp
    lt: _timestamp_pb2.Timestamp
    lte: _timestamp_pb2.Timestamp
    def __init__(self, lt: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., gt: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., gte: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., lte: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class DeleteFieldIndexCollection(_message.Message):
    __slots__ = ["collection_name", "field_name", "ordering", "wait"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FIELD_NAME_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    field_name: str
    ordering: WriteOrdering
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., field_name: _Optional[str] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ...) -> None: ...

class DeletePayloadPoints(_message.Message):
    __slots__ = ["collection_name", "keys", "ordering", "points_selector", "shard_key_selector", "wait"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    KEYS_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    POINTS_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    keys: _containers.RepeatedScalarFieldContainer[str]
    ordering: WriteOrdering
    points_selector: PointsSelector
    shard_key_selector: ShardKeySelector
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., keys: _Optional[_Iterable[str]] = ..., points_selector: _Optional[_Union[PointsSelector, _Mapping]] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class DeletePointVectors(_message.Message):
    __slots__ = ["collection_name", "ordering", "points_selector", "shard_key_selector", "vectors", "wait"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    POINTS_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    ordering: WriteOrdering
    points_selector: PointsSelector
    shard_key_selector: ShardKeySelector
    vectors: VectorsSelector
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., points_selector: _Optional[_Union[PointsSelector, _Mapping]] = ..., vectors: _Optional[_Union[VectorsSelector, _Mapping]] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class DeletePoints(_message.Message):
    __slots__ = ["collection_name", "ordering", "points", "shard_key_selector", "wait"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    ordering: WriteOrdering
    points: PointsSelector
    shard_key_selector: ShardKeySelector
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., points: _Optional[_Union[PointsSelector, _Mapping]] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class DenseVector(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, data: _Optional[_Iterable[float]] = ...) -> None: ...

class DiscoverBatchPoints(_message.Message):
    __slots__ = ["collection_name", "discover_points", "read_consistency", "timeout"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    DISCOVER_POINTS_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    discover_points: _containers.RepeatedCompositeFieldContainer[DiscoverPoints]
    read_consistency: ReadConsistency
    timeout: int
    def __init__(self, collection_name: _Optional[str] = ..., discover_points: _Optional[_Iterable[_Union[DiscoverPoints, _Mapping]]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., timeout: _Optional[int] = ...) -> None: ...

class DiscoverBatchResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[BatchResult]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[BatchResult, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class DiscoverInput(_message.Message):
    __slots__ = ["context", "target"]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    context: ContextInput
    target: VectorInput
    def __init__(self, target: _Optional[_Union[VectorInput, _Mapping]] = ..., context: _Optional[_Union[ContextInput, _Mapping]] = ...) -> None: ...

class DiscoverPoints(_message.Message):
    __slots__ = ["collection_name", "context", "filter", "limit", "lookup_from", "offset", "params", "read_consistency", "shard_key_selector", "target", "timeout", "using", "with_payload", "with_vectors"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    LOOKUP_FROM_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    USING_FIELD_NUMBER: _ClassVar[int]
    WITH_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    WITH_VECTORS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    context: _containers.RepeatedCompositeFieldContainer[ContextExamplePair]
    filter: Filter
    limit: int
    lookup_from: LookupLocation
    offset: int
    params: SearchParams
    read_consistency: ReadConsistency
    shard_key_selector: ShardKeySelector
    target: TargetVector
    timeout: int
    using: str
    with_payload: WithPayloadSelector
    with_vectors: WithVectorsSelector
    def __init__(self, collection_name: _Optional[str] = ..., target: _Optional[_Union[TargetVector, _Mapping]] = ..., context: _Optional[_Iterable[_Union[ContextExamplePair, _Mapping]]] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., limit: _Optional[int] = ..., with_payload: _Optional[_Union[WithPayloadSelector, _Mapping]] = ..., params: _Optional[_Union[SearchParams, _Mapping]] = ..., offset: _Optional[int] = ..., using: _Optional[str] = ..., with_vectors: _Optional[_Union[WithVectorsSelector, _Mapping]] = ..., lookup_from: _Optional[_Union[LookupLocation, _Mapping]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., timeout: _Optional[int] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class DiscoverResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[ScoredPoint]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[ScoredPoint, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class FieldCondition(_message.Message):
    __slots__ = ["datetime_range", "geo_bounding_box", "geo_polygon", "geo_radius", "key", "match", "range", "values_count"]
    DATETIME_RANGE_FIELD_NUMBER: _ClassVar[int]
    GEO_BOUNDING_BOX_FIELD_NUMBER: _ClassVar[int]
    GEO_POLYGON_FIELD_NUMBER: _ClassVar[int]
    GEO_RADIUS_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    MATCH_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    VALUES_COUNT_FIELD_NUMBER: _ClassVar[int]
    datetime_range: DatetimeRange
    geo_bounding_box: GeoBoundingBox
    geo_polygon: GeoPolygon
    geo_radius: GeoRadius
    key: str
    match: Match
    range: Range
    values_count: ValuesCount
    def __init__(self, key: _Optional[str] = ..., match: _Optional[_Union[Match, _Mapping]] = ..., range: _Optional[_Union[Range, _Mapping]] = ..., geo_bounding_box: _Optional[_Union[GeoBoundingBox, _Mapping]] = ..., geo_radius: _Optional[_Union[GeoRadius, _Mapping]] = ..., values_count: _Optional[_Union[ValuesCount, _Mapping]] = ..., geo_polygon: _Optional[_Union[GeoPolygon, _Mapping]] = ..., datetime_range: _Optional[_Union[DatetimeRange, _Mapping]] = ...) -> None: ...

class Filter(_message.Message):
    __slots__ = ["min_should", "must", "must_not", "should"]
    MIN_SHOULD_FIELD_NUMBER: _ClassVar[int]
    MUST_FIELD_NUMBER: _ClassVar[int]
    MUST_NOT_FIELD_NUMBER: _ClassVar[int]
    SHOULD_FIELD_NUMBER: _ClassVar[int]
    min_should: MinShould
    must: _containers.RepeatedCompositeFieldContainer[Condition]
    must_not: _containers.RepeatedCompositeFieldContainer[Condition]
    should: _containers.RepeatedCompositeFieldContainer[Condition]
    def __init__(self, should: _Optional[_Iterable[_Union[Condition, _Mapping]]] = ..., must: _Optional[_Iterable[_Union[Condition, _Mapping]]] = ..., must_not: _Optional[_Iterable[_Union[Condition, _Mapping]]] = ..., min_should: _Optional[_Union[MinShould, _Mapping]] = ...) -> None: ...

class GeoBoundingBox(_message.Message):
    __slots__ = ["bottom_right", "top_left"]
    BOTTOM_RIGHT_FIELD_NUMBER: _ClassVar[int]
    TOP_LEFT_FIELD_NUMBER: _ClassVar[int]
    bottom_right: GeoPoint
    top_left: GeoPoint
    def __init__(self, top_left: _Optional[_Union[GeoPoint, _Mapping]] = ..., bottom_right: _Optional[_Union[GeoPoint, _Mapping]] = ...) -> None: ...

class GeoLineString(_message.Message):
    __slots__ = ["points"]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    points: _containers.RepeatedCompositeFieldContainer[GeoPoint]
    def __init__(self, points: _Optional[_Iterable[_Union[GeoPoint, _Mapping]]] = ...) -> None: ...

class GeoPoint(_message.Message):
    __slots__ = ["lat", "lon"]
    LAT_FIELD_NUMBER: _ClassVar[int]
    LON_FIELD_NUMBER: _ClassVar[int]
    lat: float
    lon: float
    def __init__(self, lon: _Optional[float] = ..., lat: _Optional[float] = ...) -> None: ...

class GeoPolygon(_message.Message):
    __slots__ = ["exterior", "interiors"]
    EXTERIOR_FIELD_NUMBER: _ClassVar[int]
    INTERIORS_FIELD_NUMBER: _ClassVar[int]
    exterior: GeoLineString
    interiors: _containers.RepeatedCompositeFieldContainer[GeoLineString]
    def __init__(self, exterior: _Optional[_Union[GeoLineString, _Mapping]] = ..., interiors: _Optional[_Iterable[_Union[GeoLineString, _Mapping]]] = ...) -> None: ...

class GeoRadius(_message.Message):
    __slots__ = ["center", "radius"]
    CENTER_FIELD_NUMBER: _ClassVar[int]
    RADIUS_FIELD_NUMBER: _ClassVar[int]
    center: GeoPoint
    radius: float
    def __init__(self, center: _Optional[_Union[GeoPoint, _Mapping]] = ..., radius: _Optional[float] = ...) -> None: ...

class GetPoints(_message.Message):
    __slots__ = ["collection_name", "ids", "read_consistency", "shard_key_selector", "with_payload", "with_vectors"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    IDS_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    WITH_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    WITH_VECTORS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    ids: _containers.RepeatedCompositeFieldContainer[PointId]
    read_consistency: ReadConsistency
    shard_key_selector: ShardKeySelector
    with_payload: WithPayloadSelector
    with_vectors: WithVectorsSelector
    def __init__(self, collection_name: _Optional[str] = ..., ids: _Optional[_Iterable[_Union[PointId, _Mapping]]] = ..., with_payload: _Optional[_Union[WithPayloadSelector, _Mapping]] = ..., with_vectors: _Optional[_Union[WithVectorsSelector, _Mapping]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class GetResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[RetrievedPoint]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[RetrievedPoint, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class GroupId(_message.Message):
    __slots__ = ["integer_value", "string_value", "unsigned_value"]
    INTEGER_VALUE_FIELD_NUMBER: _ClassVar[int]
    STRING_VALUE_FIELD_NUMBER: _ClassVar[int]
    UNSIGNED_VALUE_FIELD_NUMBER: _ClassVar[int]
    integer_value: int
    string_value: str
    unsigned_value: int
    def __init__(self, unsigned_value: _Optional[int] = ..., integer_value: _Optional[int] = ..., string_value: _Optional[str] = ...) -> None: ...

class GroupsResult(_message.Message):
    __slots__ = ["groups"]
    GROUPS_FIELD_NUMBER: _ClassVar[int]
    groups: _containers.RepeatedCompositeFieldContainer[PointGroup]
    def __init__(self, groups: _Optional[_Iterable[_Union[PointGroup, _Mapping]]] = ...) -> None: ...

class HasIdCondition(_message.Message):
    __slots__ = ["has_id"]
    HAS_ID_FIELD_NUMBER: _ClassVar[int]
    has_id: _containers.RepeatedCompositeFieldContainer[PointId]
    def __init__(self, has_id: _Optional[_Iterable[_Union[PointId, _Mapping]]] = ...) -> None: ...

class IsEmptyCondition(_message.Message):
    __slots__ = ["key"]
    KEY_FIELD_NUMBER: _ClassVar[int]
    key: str
    def __init__(self, key: _Optional[str] = ...) -> None: ...

class IsNullCondition(_message.Message):
    __slots__ = ["key"]
    KEY_FIELD_NUMBER: _ClassVar[int]
    key: str
    def __init__(self, key: _Optional[str] = ...) -> None: ...

class LookupLocation(_message.Message):
    __slots__ = ["collection_name", "shard_key_selector", "vector_name"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    VECTOR_NAME_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    shard_key_selector: ShardKeySelector
    vector_name: str
    def __init__(self, collection_name: _Optional[str] = ..., vector_name: _Optional[str] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class Match(_message.Message):
    __slots__ = ["boolean", "except_integers", "except_keywords", "integer", "integers", "keyword", "keywords", "text"]
    BOOLEAN_FIELD_NUMBER: _ClassVar[int]
    EXCEPT_INTEGERS_FIELD_NUMBER: _ClassVar[int]
    EXCEPT_KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    INTEGERS_FIELD_NUMBER: _ClassVar[int]
    INTEGER_FIELD_NUMBER: _ClassVar[int]
    KEYWORDS_FIELD_NUMBER: _ClassVar[int]
    KEYWORD_FIELD_NUMBER: _ClassVar[int]
    TEXT_FIELD_NUMBER: _ClassVar[int]
    boolean: bool
    except_integers: RepeatedIntegers
    except_keywords: RepeatedStrings
    integer: int
    integers: RepeatedIntegers
    keyword: str
    keywords: RepeatedStrings
    text: str
    def __init__(self, keyword: _Optional[str] = ..., integer: _Optional[int] = ..., boolean: bool = ..., text: _Optional[str] = ..., keywords: _Optional[_Union[RepeatedStrings, _Mapping]] = ..., integers: _Optional[_Union[RepeatedIntegers, _Mapping]] = ..., except_integers: _Optional[_Union[RepeatedIntegers, _Mapping]] = ..., except_keywords: _Optional[_Union[RepeatedStrings, _Mapping]] = ...) -> None: ...

class MinShould(_message.Message):
    __slots__ = ["conditions", "min_count"]
    CONDITIONS_FIELD_NUMBER: _ClassVar[int]
    MIN_COUNT_FIELD_NUMBER: _ClassVar[int]
    conditions: _containers.RepeatedCompositeFieldContainer[Condition]
    min_count: int
    def __init__(self, conditions: _Optional[_Iterable[_Union[Condition, _Mapping]]] = ..., min_count: _Optional[int] = ...) -> None: ...

class MultiDenseVector(_message.Message):
    __slots__ = ["vectors"]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    vectors: _containers.RepeatedCompositeFieldContainer[DenseVector]
    def __init__(self, vectors: _Optional[_Iterable[_Union[DenseVector, _Mapping]]] = ...) -> None: ...

class NamedVectors(_message.Message):
    __slots__ = ["vectors"]
    class VectorsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Vector
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Vector, _Mapping]] = ...) -> None: ...
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    vectors: _containers.MessageMap[str, Vector]
    def __init__(self, vectors: _Optional[_Mapping[str, Vector]] = ...) -> None: ...

class NestedCondition(_message.Message):
    __slots__ = ["filter", "key"]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    filter: Filter
    key: str
    def __init__(self, key: _Optional[str] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ...) -> None: ...

class OrderBy(_message.Message):
    __slots__ = ["direction", "key", "start_from"]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    START_FROM_FIELD_NUMBER: _ClassVar[int]
    direction: Direction
    key: str
    start_from: StartFrom
    def __init__(self, key: _Optional[str] = ..., direction: _Optional[_Union[Direction, str]] = ..., start_from: _Optional[_Union[StartFrom, _Mapping]] = ...) -> None: ...

class OrderValue(_message.Message):
    __slots__ = ["float", "int"]
    FLOAT_FIELD_NUMBER: _ClassVar[int]
    INT_FIELD_NUMBER: _ClassVar[int]
    float: float
    int: int
    def __init__(self, int: _Optional[int] = ..., float: _Optional[float] = ...) -> None: ...

class PayloadExcludeSelector(_message.Message):
    __slots__ = ["fields"]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, fields: _Optional[_Iterable[str]] = ...) -> None: ...

class PayloadIncludeSelector(_message.Message):
    __slots__ = ["fields"]
    FIELDS_FIELD_NUMBER: _ClassVar[int]
    fields: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, fields: _Optional[_Iterable[str]] = ...) -> None: ...

class PointGroup(_message.Message):
    __slots__ = ["hits", "id", "lookup"]
    HITS_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LOOKUP_FIELD_NUMBER: _ClassVar[int]
    hits: _containers.RepeatedCompositeFieldContainer[ScoredPoint]
    id: GroupId
    lookup: RetrievedPoint
    def __init__(self, id: _Optional[_Union[GroupId, _Mapping]] = ..., hits: _Optional[_Iterable[_Union[ScoredPoint, _Mapping]]] = ..., lookup: _Optional[_Union[RetrievedPoint, _Mapping]] = ...) -> None: ...

class PointId(_message.Message):
    __slots__ = ["num", "uuid"]
    NUM_FIELD_NUMBER: _ClassVar[int]
    UUID_FIELD_NUMBER: _ClassVar[int]
    num: int
    uuid: str
    def __init__(self, num: _Optional[int] = ..., uuid: _Optional[str] = ...) -> None: ...

class PointStruct(_message.Message):
    __slots__ = ["id", "payload", "vectors"]
    class PayloadEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _json_with_int_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_json_with_int_pb2.Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    id: PointId
    payload: _containers.MessageMap[str, _json_with_int_pb2.Value]
    vectors: Vectors
    def __init__(self, id: _Optional[_Union[PointId, _Mapping]] = ..., payload: _Optional[_Mapping[str, _json_with_int_pb2.Value]] = ..., vectors: _Optional[_Union[Vectors, _Mapping]] = ...) -> None: ...

class PointVectors(_message.Message):
    __slots__ = ["id", "vectors"]
    ID_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    id: PointId
    vectors: Vectors
    def __init__(self, id: _Optional[_Union[PointId, _Mapping]] = ..., vectors: _Optional[_Union[Vectors, _Mapping]] = ...) -> None: ...

class PointsIdsList(_message.Message):
    __slots__ = ["ids"]
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedCompositeFieldContainer[PointId]
    def __init__(self, ids: _Optional[_Iterable[_Union[PointId, _Mapping]]] = ...) -> None: ...

class PointsOperationResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: UpdateResult
    time: float
    def __init__(self, result: _Optional[_Union[UpdateResult, _Mapping]] = ..., time: _Optional[float] = ...) -> None: ...

class PointsSelector(_message.Message):
    __slots__ = ["filter", "points"]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    filter: Filter
    points: PointsIdsList
    def __init__(self, points: _Optional[_Union[PointsIdsList, _Mapping]] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ...) -> None: ...

class PointsUpdateOperation(_message.Message):
    __slots__ = ["clear_payload", "clear_payload_deprecated", "delete_deprecated", "delete_payload", "delete_points", "delete_vectors", "overwrite_payload", "set_payload", "update_vectors", "upsert"]
    class ClearPayload(_message.Message):
        __slots__ = ["points", "shard_key_selector"]
        POINTS_FIELD_NUMBER: _ClassVar[int]
        SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        points: PointsSelector
        shard_key_selector: ShardKeySelector
        def __init__(self, points: _Optional[_Union[PointsSelector, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...
    class DeletePayload(_message.Message):
        __slots__ = ["keys", "points_selector", "shard_key_selector"]
        KEYS_FIELD_NUMBER: _ClassVar[int]
        POINTS_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        keys: _containers.RepeatedScalarFieldContainer[str]
        points_selector: PointsSelector
        shard_key_selector: ShardKeySelector
        def __init__(self, keys: _Optional[_Iterable[str]] = ..., points_selector: _Optional[_Union[PointsSelector, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...
    class DeletePoints(_message.Message):
        __slots__ = ["points", "shard_key_selector"]
        POINTS_FIELD_NUMBER: _ClassVar[int]
        SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        points: PointsSelector
        shard_key_selector: ShardKeySelector
        def __init__(self, points: _Optional[_Union[PointsSelector, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...
    class DeleteVectors(_message.Message):
        __slots__ = ["points_selector", "shard_key_selector", "vectors"]
        POINTS_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        VECTORS_FIELD_NUMBER: _ClassVar[int]
        points_selector: PointsSelector
        shard_key_selector: ShardKeySelector
        vectors: VectorsSelector
        def __init__(self, points_selector: _Optional[_Union[PointsSelector, _Mapping]] = ..., vectors: _Optional[_Union[VectorsSelector, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...
    class OverwritePayload(_message.Message):
        __slots__ = ["key", "payload", "points_selector", "shard_key_selector"]
        class PayloadEntry(_message.Message):
            __slots__ = ["key", "value"]
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: _json_with_int_pb2.Value
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_json_with_int_pb2.Value, _Mapping]] = ...) -> None: ...
        KEY_FIELD_NUMBER: _ClassVar[int]
        PAYLOAD_FIELD_NUMBER: _ClassVar[int]
        POINTS_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        key: str
        payload: _containers.MessageMap[str, _json_with_int_pb2.Value]
        points_selector: PointsSelector
        shard_key_selector: ShardKeySelector
        def __init__(self, payload: _Optional[_Mapping[str, _json_with_int_pb2.Value]] = ..., points_selector: _Optional[_Union[PointsSelector, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ..., key: _Optional[str] = ...) -> None: ...
    class PointStructList(_message.Message):
        __slots__ = ["points", "shard_key_selector"]
        POINTS_FIELD_NUMBER: _ClassVar[int]
        SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        points: _containers.RepeatedCompositeFieldContainer[PointStruct]
        shard_key_selector: ShardKeySelector
        def __init__(self, points: _Optional[_Iterable[_Union[PointStruct, _Mapping]]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...
    class SetPayload(_message.Message):
        __slots__ = ["key", "payload", "points_selector", "shard_key_selector"]
        class PayloadEntry(_message.Message):
            __slots__ = ["key", "value"]
            KEY_FIELD_NUMBER: _ClassVar[int]
            VALUE_FIELD_NUMBER: _ClassVar[int]
            key: str
            value: _json_with_int_pb2.Value
            def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_json_with_int_pb2.Value, _Mapping]] = ...) -> None: ...
        KEY_FIELD_NUMBER: _ClassVar[int]
        PAYLOAD_FIELD_NUMBER: _ClassVar[int]
        POINTS_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        key: str
        payload: _containers.MessageMap[str, _json_with_int_pb2.Value]
        points_selector: PointsSelector
        shard_key_selector: ShardKeySelector
        def __init__(self, payload: _Optional[_Mapping[str, _json_with_int_pb2.Value]] = ..., points_selector: _Optional[_Union[PointsSelector, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ..., key: _Optional[str] = ...) -> None: ...
    class UpdateVectors(_message.Message):
        __slots__ = ["points", "shard_key_selector"]
        POINTS_FIELD_NUMBER: _ClassVar[int]
        SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
        points: _containers.RepeatedCompositeFieldContainer[PointVectors]
        shard_key_selector: ShardKeySelector
        def __init__(self, points: _Optional[_Iterable[_Union[PointVectors, _Mapping]]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...
    CLEAR_PAYLOAD_DEPRECATED_FIELD_NUMBER: _ClassVar[int]
    CLEAR_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    DELETE_DEPRECATED_FIELD_NUMBER: _ClassVar[int]
    DELETE_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    DELETE_POINTS_FIELD_NUMBER: _ClassVar[int]
    DELETE_VECTORS_FIELD_NUMBER: _ClassVar[int]
    OVERWRITE_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    SET_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    UPDATE_VECTORS_FIELD_NUMBER: _ClassVar[int]
    UPSERT_FIELD_NUMBER: _ClassVar[int]
    clear_payload: PointsUpdateOperation.ClearPayload
    clear_payload_deprecated: PointsSelector
    delete_deprecated: PointsSelector
    delete_payload: PointsUpdateOperation.DeletePayload
    delete_points: PointsUpdateOperation.DeletePoints
    delete_vectors: PointsUpdateOperation.DeleteVectors
    overwrite_payload: PointsUpdateOperation.OverwritePayload
    set_payload: PointsUpdateOperation.SetPayload
    update_vectors: PointsUpdateOperation.UpdateVectors
    upsert: PointsUpdateOperation.PointStructList
    def __init__(self, upsert: _Optional[_Union[PointsUpdateOperation.PointStructList, _Mapping]] = ..., delete_deprecated: _Optional[_Union[PointsSelector, _Mapping]] = ..., set_payload: _Optional[_Union[PointsUpdateOperation.SetPayload, _Mapping]] = ..., overwrite_payload: _Optional[_Union[PointsUpdateOperation.OverwritePayload, _Mapping]] = ..., delete_payload: _Optional[_Union[PointsUpdateOperation.DeletePayload, _Mapping]] = ..., clear_payload_deprecated: _Optional[_Union[PointsSelector, _Mapping]] = ..., update_vectors: _Optional[_Union[PointsUpdateOperation.UpdateVectors, _Mapping]] = ..., delete_vectors: _Optional[_Union[PointsUpdateOperation.DeleteVectors, _Mapping]] = ..., delete_points: _Optional[_Union[PointsUpdateOperation.DeletePoints, _Mapping]] = ..., clear_payload: _Optional[_Union[PointsUpdateOperation.ClearPayload, _Mapping]] = ...) -> None: ...

class PrefetchQuery(_message.Message):
    __slots__ = ["filter", "limit", "lookup_from", "params", "prefetch", "query", "score_threshold", "using"]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    LOOKUP_FROM_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    PREFETCH_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    USING_FIELD_NUMBER: _ClassVar[int]
    filter: Filter
    limit: int
    lookup_from: LookupLocation
    params: SearchParams
    prefetch: _containers.RepeatedCompositeFieldContainer[PrefetchQuery]
    query: Query
    score_threshold: float
    using: str
    def __init__(self, prefetch: _Optional[_Iterable[_Union[PrefetchQuery, _Mapping]]] = ..., query: _Optional[_Union[Query, _Mapping]] = ..., using: _Optional[str] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., params: _Optional[_Union[SearchParams, _Mapping]] = ..., score_threshold: _Optional[float] = ..., limit: _Optional[int] = ..., lookup_from: _Optional[_Union[LookupLocation, _Mapping]] = ...) -> None: ...

class QuantizationSearchParams(_message.Message):
    __slots__ = ["ignore", "oversampling", "rescore"]
    IGNORE_FIELD_NUMBER: _ClassVar[int]
    OVERSAMPLING_FIELD_NUMBER: _ClassVar[int]
    RESCORE_FIELD_NUMBER: _ClassVar[int]
    ignore: bool
    oversampling: float
    rescore: bool
    def __init__(self, ignore: bool = ..., rescore: bool = ..., oversampling: _Optional[float] = ...) -> None: ...

class Query(_message.Message):
    __slots__ = ["context", "discover", "fusion", "nearest", "order_by", "recommend"]
    CONTEXT_FIELD_NUMBER: _ClassVar[int]
    DISCOVER_FIELD_NUMBER: _ClassVar[int]
    FUSION_FIELD_NUMBER: _ClassVar[int]
    NEAREST_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    RECOMMEND_FIELD_NUMBER: _ClassVar[int]
    context: ContextInput
    discover: DiscoverInput
    fusion: Fusion
    nearest: VectorInput
    order_by: OrderBy
    recommend: RecommendInput
    def __init__(self, nearest: _Optional[_Union[VectorInput, _Mapping]] = ..., recommend: _Optional[_Union[RecommendInput, _Mapping]] = ..., discover: _Optional[_Union[DiscoverInput, _Mapping]] = ..., context: _Optional[_Union[ContextInput, _Mapping]] = ..., order_by: _Optional[_Union[OrderBy, _Mapping]] = ..., fusion: _Optional[_Union[Fusion, str]] = ...) -> None: ...

class QueryBatchPoints(_message.Message):
    __slots__ = ["collection_name", "query_points", "read_consistency", "timeout"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    QUERY_POINTS_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    query_points: _containers.RepeatedCompositeFieldContainer[QueryPoints]
    read_consistency: ReadConsistency
    timeout: int
    def __init__(self, collection_name: _Optional[str] = ..., query_points: _Optional[_Iterable[_Union[QueryPoints, _Mapping]]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., timeout: _Optional[int] = ...) -> None: ...

class QueryBatchResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[BatchResult]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[BatchResult, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class QueryPoints(_message.Message):
    __slots__ = ["collection_name", "filter", "limit", "lookup_from", "offset", "params", "prefetch", "query", "read_consistency", "score_threshold", "shard_key_selector", "timeout", "using", "with_payload", "with_vectors"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    LOOKUP_FROM_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    PREFETCH_FIELD_NUMBER: _ClassVar[int]
    QUERY_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    USING_FIELD_NUMBER: _ClassVar[int]
    WITH_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    WITH_VECTORS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    filter: Filter
    limit: int
    lookup_from: LookupLocation
    offset: int
    params: SearchParams
    prefetch: _containers.RepeatedCompositeFieldContainer[PrefetchQuery]
    query: Query
    read_consistency: ReadConsistency
    score_threshold: float
    shard_key_selector: ShardKeySelector
    timeout: int
    using: str
    with_payload: WithPayloadSelector
    with_vectors: WithVectorsSelector
    def __init__(self, collection_name: _Optional[str] = ..., prefetch: _Optional[_Iterable[_Union[PrefetchQuery, _Mapping]]] = ..., query: _Optional[_Union[Query, _Mapping]] = ..., using: _Optional[str] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., params: _Optional[_Union[SearchParams, _Mapping]] = ..., score_threshold: _Optional[float] = ..., limit: _Optional[int] = ..., offset: _Optional[int] = ..., with_vectors: _Optional[_Union[WithVectorsSelector, _Mapping]] = ..., with_payload: _Optional[_Union[WithPayloadSelector, _Mapping]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ..., lookup_from: _Optional[_Union[LookupLocation, _Mapping]] = ..., timeout: _Optional[int] = ...) -> None: ...

class QueryResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[ScoredPoint]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[ScoredPoint, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class Range(_message.Message):
    __slots__ = ["gt", "gte", "lt", "lte"]
    GTE_FIELD_NUMBER: _ClassVar[int]
    GT_FIELD_NUMBER: _ClassVar[int]
    LTE_FIELD_NUMBER: _ClassVar[int]
    LT_FIELD_NUMBER: _ClassVar[int]
    gt: float
    gte: float
    lt: float
    lte: float
    def __init__(self, lt: _Optional[float] = ..., gt: _Optional[float] = ..., gte: _Optional[float] = ..., lte: _Optional[float] = ...) -> None: ...

class ReadConsistency(_message.Message):
    __slots__ = ["factor", "type"]
    FACTOR_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    factor: int
    type: ReadConsistencyType
    def __init__(self, type: _Optional[_Union[ReadConsistencyType, str]] = ..., factor: _Optional[int] = ...) -> None: ...

class RecommendBatchPoints(_message.Message):
    __slots__ = ["collection_name", "read_consistency", "recommend_points", "timeout"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    RECOMMEND_POINTS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    read_consistency: ReadConsistency
    recommend_points: _containers.RepeatedCompositeFieldContainer[RecommendPoints]
    timeout: int
    def __init__(self, collection_name: _Optional[str] = ..., recommend_points: _Optional[_Iterable[_Union[RecommendPoints, _Mapping]]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., timeout: _Optional[int] = ...) -> None: ...

class RecommendBatchResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[BatchResult]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[BatchResult, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class RecommendGroupsResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: GroupsResult
    time: float
    def __init__(self, result: _Optional[_Union[GroupsResult, _Mapping]] = ..., time: _Optional[float] = ...) -> None: ...

class RecommendInput(_message.Message):
    __slots__ = ["negative", "positive", "strategy"]
    NEGATIVE_FIELD_NUMBER: _ClassVar[int]
    POSITIVE_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    negative: _containers.RepeatedCompositeFieldContainer[VectorInput]
    positive: _containers.RepeatedCompositeFieldContainer[VectorInput]
    strategy: RecommendStrategy
    def __init__(self, positive: _Optional[_Iterable[_Union[VectorInput, _Mapping]]] = ..., negative: _Optional[_Iterable[_Union[VectorInput, _Mapping]]] = ..., strategy: _Optional[_Union[RecommendStrategy, str]] = ...) -> None: ...

class RecommendPointGroups(_message.Message):
    __slots__ = ["collection_name", "filter", "group_by", "group_size", "limit", "lookup_from", "negative", "negative_vectors", "params", "positive", "positive_vectors", "read_consistency", "score_threshold", "shard_key_selector", "strategy", "timeout", "using", "with_lookup", "with_payload", "with_vectors"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    GROUP_BY_FIELD_NUMBER: _ClassVar[int]
    GROUP_SIZE_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    LOOKUP_FROM_FIELD_NUMBER: _ClassVar[int]
    NEGATIVE_FIELD_NUMBER: _ClassVar[int]
    NEGATIVE_VECTORS_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    POSITIVE_FIELD_NUMBER: _ClassVar[int]
    POSITIVE_VECTORS_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    USING_FIELD_NUMBER: _ClassVar[int]
    WITH_LOOKUP_FIELD_NUMBER: _ClassVar[int]
    WITH_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    WITH_VECTORS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    filter: Filter
    group_by: str
    group_size: int
    limit: int
    lookup_from: LookupLocation
    negative: _containers.RepeatedCompositeFieldContainer[PointId]
    negative_vectors: _containers.RepeatedCompositeFieldContainer[Vector]
    params: SearchParams
    positive: _containers.RepeatedCompositeFieldContainer[PointId]
    positive_vectors: _containers.RepeatedCompositeFieldContainer[Vector]
    read_consistency: ReadConsistency
    score_threshold: float
    shard_key_selector: ShardKeySelector
    strategy: RecommendStrategy
    timeout: int
    using: str
    with_lookup: WithLookup
    with_payload: WithPayloadSelector
    with_vectors: WithVectorsSelector
    def __init__(self, collection_name: _Optional[str] = ..., positive: _Optional[_Iterable[_Union[PointId, _Mapping]]] = ..., negative: _Optional[_Iterable[_Union[PointId, _Mapping]]] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., limit: _Optional[int] = ..., with_payload: _Optional[_Union[WithPayloadSelector, _Mapping]] = ..., params: _Optional[_Union[SearchParams, _Mapping]] = ..., score_threshold: _Optional[float] = ..., using: _Optional[str] = ..., with_vectors: _Optional[_Union[WithVectorsSelector, _Mapping]] = ..., lookup_from: _Optional[_Union[LookupLocation, _Mapping]] = ..., group_by: _Optional[str] = ..., group_size: _Optional[int] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., with_lookup: _Optional[_Union[WithLookup, _Mapping]] = ..., strategy: _Optional[_Union[RecommendStrategy, str]] = ..., positive_vectors: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ..., negative_vectors: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ..., timeout: _Optional[int] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class RecommendPoints(_message.Message):
    __slots__ = ["collection_name", "filter", "limit", "lookup_from", "negative", "negative_vectors", "offset", "params", "positive", "positive_vectors", "read_consistency", "score_threshold", "shard_key_selector", "strategy", "timeout", "using", "with_payload", "with_vectors"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    LOOKUP_FROM_FIELD_NUMBER: _ClassVar[int]
    NEGATIVE_FIELD_NUMBER: _ClassVar[int]
    NEGATIVE_VECTORS_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    POSITIVE_FIELD_NUMBER: _ClassVar[int]
    POSITIVE_VECTORS_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    STRATEGY_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    USING_FIELD_NUMBER: _ClassVar[int]
    WITH_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    WITH_VECTORS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    filter: Filter
    limit: int
    lookup_from: LookupLocation
    negative: _containers.RepeatedCompositeFieldContainer[PointId]
    negative_vectors: _containers.RepeatedCompositeFieldContainer[Vector]
    offset: int
    params: SearchParams
    positive: _containers.RepeatedCompositeFieldContainer[PointId]
    positive_vectors: _containers.RepeatedCompositeFieldContainer[Vector]
    read_consistency: ReadConsistency
    score_threshold: float
    shard_key_selector: ShardKeySelector
    strategy: RecommendStrategy
    timeout: int
    using: str
    with_payload: WithPayloadSelector
    with_vectors: WithVectorsSelector
    def __init__(self, collection_name: _Optional[str] = ..., positive: _Optional[_Iterable[_Union[PointId, _Mapping]]] = ..., negative: _Optional[_Iterable[_Union[PointId, _Mapping]]] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., limit: _Optional[int] = ..., with_payload: _Optional[_Union[WithPayloadSelector, _Mapping]] = ..., params: _Optional[_Union[SearchParams, _Mapping]] = ..., score_threshold: _Optional[float] = ..., offset: _Optional[int] = ..., using: _Optional[str] = ..., with_vectors: _Optional[_Union[WithVectorsSelector, _Mapping]] = ..., lookup_from: _Optional[_Union[LookupLocation, _Mapping]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., strategy: _Optional[_Union[RecommendStrategy, str]] = ..., positive_vectors: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ..., negative_vectors: _Optional[_Iterable[_Union[Vector, _Mapping]]] = ..., timeout: _Optional[int] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class RecommendResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[ScoredPoint]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[ScoredPoint, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class RepeatedIntegers(_message.Message):
    __slots__ = ["integers"]
    INTEGERS_FIELD_NUMBER: _ClassVar[int]
    integers: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, integers: _Optional[_Iterable[int]] = ...) -> None: ...

class RepeatedStrings(_message.Message):
    __slots__ = ["strings"]
    STRINGS_FIELD_NUMBER: _ClassVar[int]
    strings: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, strings: _Optional[_Iterable[str]] = ...) -> None: ...

class RetrievedPoint(_message.Message):
    __slots__ = ["id", "order_value", "payload", "shard_key", "vectors"]
    class PayloadEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _json_with_int_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_json_with_int_pb2.Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    ORDER_VALUE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    id: PointId
    order_value: OrderValue
    payload: _containers.MessageMap[str, _json_with_int_pb2.Value]
    shard_key: _collections_pb2.ShardKey
    vectors: Vectors
    def __init__(self, id: _Optional[_Union[PointId, _Mapping]] = ..., payload: _Optional[_Mapping[str, _json_with_int_pb2.Value]] = ..., vectors: _Optional[_Union[Vectors, _Mapping]] = ..., shard_key: _Optional[_Union[_collections_pb2.ShardKey, _Mapping]] = ..., order_value: _Optional[_Union[OrderValue, _Mapping]] = ...) -> None: ...

class ScoredPoint(_message.Message):
    __slots__ = ["id", "order_value", "payload", "score", "shard_key", "vectors", "version"]
    class PayloadEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _json_with_int_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_json_with_int_pb2.Value, _Mapping]] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    ORDER_VALUE_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    SCORE_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_FIELD_NUMBER: _ClassVar[int]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    id: PointId
    order_value: OrderValue
    payload: _containers.MessageMap[str, _json_with_int_pb2.Value]
    score: float
    shard_key: _collections_pb2.ShardKey
    vectors: Vectors
    version: int
    def __init__(self, id: _Optional[_Union[PointId, _Mapping]] = ..., payload: _Optional[_Mapping[str, _json_with_int_pb2.Value]] = ..., score: _Optional[float] = ..., version: _Optional[int] = ..., vectors: _Optional[_Union[Vectors, _Mapping]] = ..., shard_key: _Optional[_Union[_collections_pb2.ShardKey, _Mapping]] = ..., order_value: _Optional[_Union[OrderValue, _Mapping]] = ...) -> None: ...

class ScrollPoints(_message.Message):
    __slots__ = ["collection_name", "filter", "limit", "offset", "order_by", "read_consistency", "shard_key_selector", "with_payload", "with_vectors"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    ORDER_BY_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    WITH_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    WITH_VECTORS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    filter: Filter
    limit: int
    offset: PointId
    order_by: OrderBy
    read_consistency: ReadConsistency
    shard_key_selector: ShardKeySelector
    with_payload: WithPayloadSelector
    with_vectors: WithVectorsSelector
    def __init__(self, collection_name: _Optional[str] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., offset: _Optional[_Union[PointId, _Mapping]] = ..., limit: _Optional[int] = ..., with_payload: _Optional[_Union[WithPayloadSelector, _Mapping]] = ..., with_vectors: _Optional[_Union[WithVectorsSelector, _Mapping]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ..., order_by: _Optional[_Union[OrderBy, _Mapping]] = ...) -> None: ...

class ScrollResponse(_message.Message):
    __slots__ = ["next_page_offset", "result", "time"]
    NEXT_PAGE_OFFSET_FIELD_NUMBER: _ClassVar[int]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    next_page_offset: PointId
    result: _containers.RepeatedCompositeFieldContainer[RetrievedPoint]
    time: float
    def __init__(self, next_page_offset: _Optional[_Union[PointId, _Mapping]] = ..., result: _Optional[_Iterable[_Union[RetrievedPoint, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class SearchBatchPoints(_message.Message):
    __slots__ = ["collection_name", "read_consistency", "search_points", "timeout"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SEARCH_POINTS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    read_consistency: ReadConsistency
    search_points: _containers.RepeatedCompositeFieldContainer[SearchPoints]
    timeout: int
    def __init__(self, collection_name: _Optional[str] = ..., search_points: _Optional[_Iterable[_Union[SearchPoints, _Mapping]]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., timeout: _Optional[int] = ...) -> None: ...

class SearchBatchResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[BatchResult]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[BatchResult, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class SearchGroupsResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: GroupsResult
    time: float
    def __init__(self, result: _Optional[_Union[GroupsResult, _Mapping]] = ..., time: _Optional[float] = ...) -> None: ...

class SearchParams(_message.Message):
    __slots__ = ["exact", "hnsw_ef", "indexed_only", "quantization"]
    EXACT_FIELD_NUMBER: _ClassVar[int]
    HNSW_EF_FIELD_NUMBER: _ClassVar[int]
    INDEXED_ONLY_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_FIELD_NUMBER: _ClassVar[int]
    exact: bool
    hnsw_ef: int
    indexed_only: bool
    quantization: QuantizationSearchParams
    def __init__(self, hnsw_ef: _Optional[int] = ..., exact: bool = ..., quantization: _Optional[_Union[QuantizationSearchParams, _Mapping]] = ..., indexed_only: bool = ...) -> None: ...

class SearchPointGroups(_message.Message):
    __slots__ = ["collection_name", "filter", "group_by", "group_size", "limit", "params", "read_consistency", "score_threshold", "shard_key_selector", "sparse_indices", "timeout", "vector", "vector_name", "with_lookup", "with_payload", "with_vectors"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    GROUP_BY_FIELD_NUMBER: _ClassVar[int]
    GROUP_SIZE_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    SPARSE_INDICES_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    VECTOR_NAME_FIELD_NUMBER: _ClassVar[int]
    WITH_LOOKUP_FIELD_NUMBER: _ClassVar[int]
    WITH_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    WITH_VECTORS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    filter: Filter
    group_by: str
    group_size: int
    limit: int
    params: SearchParams
    read_consistency: ReadConsistency
    score_threshold: float
    shard_key_selector: ShardKeySelector
    sparse_indices: SparseIndices
    timeout: int
    vector: _containers.RepeatedScalarFieldContainer[float]
    vector_name: str
    with_lookup: WithLookup
    with_payload: WithPayloadSelector
    with_vectors: WithVectorsSelector
    def __init__(self, collection_name: _Optional[str] = ..., vector: _Optional[_Iterable[float]] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., limit: _Optional[int] = ..., with_payload: _Optional[_Union[WithPayloadSelector, _Mapping]] = ..., params: _Optional[_Union[SearchParams, _Mapping]] = ..., score_threshold: _Optional[float] = ..., vector_name: _Optional[str] = ..., with_vectors: _Optional[_Union[WithVectorsSelector, _Mapping]] = ..., group_by: _Optional[str] = ..., group_size: _Optional[int] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., with_lookup: _Optional[_Union[WithLookup, _Mapping]] = ..., timeout: _Optional[int] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ..., sparse_indices: _Optional[_Union[SparseIndices, _Mapping]] = ...) -> None: ...

class SearchPoints(_message.Message):
    __slots__ = ["collection_name", "filter", "limit", "offset", "params", "read_consistency", "score_threshold", "shard_key_selector", "sparse_indices", "timeout", "vector", "vector_name", "with_payload", "with_vectors"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    FILTER_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    READ_CONSISTENCY_FIELD_NUMBER: _ClassVar[int]
    SCORE_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    SPARSE_INDICES_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    VECTOR_NAME_FIELD_NUMBER: _ClassVar[int]
    WITH_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    WITH_VECTORS_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    filter: Filter
    limit: int
    offset: int
    params: SearchParams
    read_consistency: ReadConsistency
    score_threshold: float
    shard_key_selector: ShardKeySelector
    sparse_indices: SparseIndices
    timeout: int
    vector: _containers.RepeatedScalarFieldContainer[float]
    vector_name: str
    with_payload: WithPayloadSelector
    with_vectors: WithVectorsSelector
    def __init__(self, collection_name: _Optional[str] = ..., vector: _Optional[_Iterable[float]] = ..., filter: _Optional[_Union[Filter, _Mapping]] = ..., limit: _Optional[int] = ..., with_payload: _Optional[_Union[WithPayloadSelector, _Mapping]] = ..., params: _Optional[_Union[SearchParams, _Mapping]] = ..., score_threshold: _Optional[float] = ..., offset: _Optional[int] = ..., vector_name: _Optional[str] = ..., with_vectors: _Optional[_Union[WithVectorsSelector, _Mapping]] = ..., read_consistency: _Optional[_Union[ReadConsistency, _Mapping]] = ..., timeout: _Optional[int] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ..., sparse_indices: _Optional[_Union[SparseIndices, _Mapping]] = ...) -> None: ...

class SearchResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[ScoredPoint]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[ScoredPoint, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class SetPayloadPoints(_message.Message):
    __slots__ = ["collection_name", "key", "ordering", "payload", "points_selector", "shard_key_selector", "wait"]
    class PayloadEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: _json_with_int_pb2.Value
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[_json_with_int_pb2.Value, _Mapping]] = ...) -> None: ...
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    POINTS_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    key: str
    ordering: WriteOrdering
    payload: _containers.MessageMap[str, _json_with_int_pb2.Value]
    points_selector: PointsSelector
    shard_key_selector: ShardKeySelector
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., payload: _Optional[_Mapping[str, _json_with_int_pb2.Value]] = ..., points_selector: _Optional[_Union[PointsSelector, _Mapping]] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ..., key: _Optional[str] = ...) -> None: ...

class ShardKeySelector(_message.Message):
    __slots__ = ["shard_keys"]
    SHARD_KEYS_FIELD_NUMBER: _ClassVar[int]
    shard_keys: _containers.RepeatedCompositeFieldContainer[_collections_pb2.ShardKey]
    def __init__(self, shard_keys: _Optional[_Iterable[_Union[_collections_pb2.ShardKey, _Mapping]]] = ...) -> None: ...

class SparseIndices(_message.Message):
    __slots__ = ["data"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, data: _Optional[_Iterable[int]] = ...) -> None: ...

class SparseVector(_message.Message):
    __slots__ = ["indices", "values"]
    INDICES_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    indices: _containers.RepeatedScalarFieldContainer[int]
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, values: _Optional[_Iterable[float]] = ..., indices: _Optional[_Iterable[int]] = ...) -> None: ...

class StartFrom(_message.Message):
    __slots__ = ["datetime", "float", "integer", "timestamp"]
    DATETIME_FIELD_NUMBER: _ClassVar[int]
    FLOAT_FIELD_NUMBER: _ClassVar[int]
    INTEGER_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    datetime: str
    float: float
    integer: int
    timestamp: _timestamp_pb2.Timestamp
    def __init__(self, float: _Optional[float] = ..., integer: _Optional[int] = ..., timestamp: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., datetime: _Optional[str] = ...) -> None: ...

class TargetVector(_message.Message):
    __slots__ = ["single"]
    SINGLE_FIELD_NUMBER: _ClassVar[int]
    single: VectorExample
    def __init__(self, single: _Optional[_Union[VectorExample, _Mapping]] = ...) -> None: ...

class UpdateBatchPoints(_message.Message):
    __slots__ = ["collection_name", "operations", "ordering", "wait"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    OPERATIONS_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    operations: _containers.RepeatedCompositeFieldContainer[PointsUpdateOperation]
    ordering: WriteOrdering
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., operations: _Optional[_Iterable[_Union[PointsUpdateOperation, _Mapping]]] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ...) -> None: ...

class UpdateBatchResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: _containers.RepeatedCompositeFieldContainer[UpdateResult]
    time: float
    def __init__(self, result: _Optional[_Iterable[_Union[UpdateResult, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class UpdatePointVectors(_message.Message):
    __slots__ = ["collection_name", "ordering", "points", "shard_key_selector", "wait"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    ordering: WriteOrdering
    points: _containers.RepeatedCompositeFieldContainer[PointVectors]
    shard_key_selector: ShardKeySelector
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., points: _Optional[_Iterable[_Union[PointVectors, _Mapping]]] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class UpdateResult(_message.Message):
    __slots__ = ["operation_id", "status"]
    OPERATION_ID_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    operation_id: int
    status: UpdateStatus
    def __init__(self, operation_id: _Optional[int] = ..., status: _Optional[_Union[UpdateStatus, str]] = ...) -> None: ...

class UpsertPoints(_message.Message):
    __slots__ = ["collection_name", "ordering", "points", "shard_key_selector", "wait"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    ORDERING_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_SELECTOR_FIELD_NUMBER: _ClassVar[int]
    WAIT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    ordering: WriteOrdering
    points: _containers.RepeatedCompositeFieldContainer[PointStruct]
    shard_key_selector: ShardKeySelector
    wait: bool
    def __init__(self, collection_name: _Optional[str] = ..., wait: bool = ..., points: _Optional[_Iterable[_Union[PointStruct, _Mapping]]] = ..., ordering: _Optional[_Union[WriteOrdering, _Mapping]] = ..., shard_key_selector: _Optional[_Union[ShardKeySelector, _Mapping]] = ...) -> None: ...

class ValuesCount(_message.Message):
    __slots__ = ["gt", "gte", "lt", "lte"]
    GTE_FIELD_NUMBER: _ClassVar[int]
    GT_FIELD_NUMBER: _ClassVar[int]
    LTE_FIELD_NUMBER: _ClassVar[int]
    LT_FIELD_NUMBER: _ClassVar[int]
    gt: int
    gte: int
    lt: int
    lte: int
    def __init__(self, lt: _Optional[int] = ..., gt: _Optional[int] = ..., gte: _Optional[int] = ..., lte: _Optional[int] = ...) -> None: ...

class Vector(_message.Message):
    __slots__ = ["data", "indices", "vectors_count"]
    DATA_FIELD_NUMBER: _ClassVar[int]
    INDICES_FIELD_NUMBER: _ClassVar[int]
    VECTORS_COUNT_FIELD_NUMBER: _ClassVar[int]
    data: _containers.RepeatedScalarFieldContainer[float]
    indices: SparseIndices
    vectors_count: int
    def __init__(self, data: _Optional[_Iterable[float]] = ..., indices: _Optional[_Union[SparseIndices, _Mapping]] = ..., vectors_count: _Optional[int] = ...) -> None: ...

class VectorExample(_message.Message):
    __slots__ = ["id", "vector"]
    ID_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    id: PointId
    vector: Vector
    def __init__(self, id: _Optional[_Union[PointId, _Mapping]] = ..., vector: _Optional[_Union[Vector, _Mapping]] = ...) -> None: ...

class VectorInput(_message.Message):
    __slots__ = ["dense", "id", "multi_dense", "sparse"]
    DENSE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    MULTI_DENSE_FIELD_NUMBER: _ClassVar[int]
    SPARSE_FIELD_NUMBER: _ClassVar[int]
    dense: DenseVector
    id: PointId
    multi_dense: MultiDenseVector
    sparse: SparseVector
    def __init__(self, id: _Optional[_Union[PointId, _Mapping]] = ..., dense: _Optional[_Union[DenseVector, _Mapping]] = ..., sparse: _Optional[_Union[SparseVector, _Mapping]] = ..., multi_dense: _Optional[_Union[MultiDenseVector, _Mapping]] = ...) -> None: ...

class Vectors(_message.Message):
    __slots__ = ["vector", "vectors"]
    VECTORS_FIELD_NUMBER: _ClassVar[int]
    VECTOR_FIELD_NUMBER: _ClassVar[int]
    vector: Vector
    vectors: NamedVectors
    def __init__(self, vector: _Optional[_Union[Vector, _Mapping]] = ..., vectors: _Optional[_Union[NamedVectors, _Mapping]] = ...) -> None: ...

class VectorsSelector(_message.Message):
    __slots__ = ["names"]
    NAMES_FIELD_NUMBER: _ClassVar[int]
    names: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, names: _Optional[_Iterable[str]] = ...) -> None: ...

class WithLookup(_message.Message):
    __slots__ = ["collection", "with_payload", "with_vectors"]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    WITH_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    WITH_VECTORS_FIELD_NUMBER: _ClassVar[int]
    collection: str
    with_payload: WithPayloadSelector
    with_vectors: WithVectorsSelector
    def __init__(self, collection: _Optional[str] = ..., with_payload: _Optional[_Union[WithPayloadSelector, _Mapping]] = ..., with_vectors: _Optional[_Union[WithVectorsSelector, _Mapping]] = ...) -> None: ...

class WithPayloadSelector(_message.Message):
    __slots__ = ["enable", "exclude", "include"]
    ENABLE_FIELD_NUMBER: _ClassVar[int]
    EXCLUDE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_FIELD_NUMBER: _ClassVar[int]
    enable: bool
    exclude: PayloadExcludeSelector
    include: PayloadIncludeSelector
    def __init__(self, enable: bool = ..., include: _Optional[_Union[PayloadIncludeSelector, _Mapping]] = ..., exclude: _Optional[_Union[PayloadExcludeSelector, _Mapping]] = ...) -> None: ...

class WithVectorsSelector(_message.Message):
    __slots__ = ["enable", "include"]
    ENABLE_FIELD_NUMBER: _ClassVar[int]
    INCLUDE_FIELD_NUMBER: _ClassVar[int]
    enable: bool
    include: VectorsSelector
    def __init__(self, enable: bool = ..., include: _Optional[_Union[VectorsSelector, _Mapping]] = ...) -> None: ...

class WriteOrdering(_message.Message):
    __slots__ = ["type"]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    type: WriteOrderingType
    def __init__(self, type: _Optional[_Union[WriteOrderingType, str]] = ...) -> None: ...

class WriteOrderingType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class ReadConsistencyType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class FieldType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class Direction(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class RecommendStrategy(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class Fusion(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class UpdateStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
