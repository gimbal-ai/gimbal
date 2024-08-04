from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

Active: ReplicaState
Auto: ShardingMethod
Bool: PayloadSchemaType
Cosine: Distance
Custom: ShardingMethod
DESCRIPTOR: _descriptor.FileDescriptor
Datetime: PayloadSchemaType
Dead: ReplicaState
Default: Datatype
Dot: Distance
Euclid: Distance
Float: PayloadSchemaType
Float16: Datatype
Float32: Datatype
Geo: PayloadSchemaType
Green: CollectionStatus
Grey: CollectionStatus
Idf: Modifier
Initializing: ReplicaState
Int8: QuantizationType
Integer: PayloadSchemaType
Keyword: PayloadSchemaType
Listener: ReplicaState
Manhattan: Distance
MaxSim: MultiVectorComparator
Multilingual: TokenizerType
None: Modifier
Partial: ReplicaState
PartialSnapshot: ReplicaState
Prefix: TokenizerType
Recovery: ReplicaState
Red: CollectionStatus
Resharding: ReplicaState
ReshardingStreamRecords: ShardTransferMethod
Snapshot: ShardTransferMethod
StreamRecords: ShardTransferMethod
Text: PayloadSchemaType
Uint8: Datatype
Unknown: TokenizerType
UnknownCollectionStatus: CollectionStatus
UnknownDistance: Distance
UnknownQuantization: QuantizationType
UnknownType: PayloadSchemaType
WalDelta: ShardTransferMethod
Whitespace: TokenizerType
Word: TokenizerType
Yellow: CollectionStatus
x16: CompressionRatio
x32: CompressionRatio
x4: CompressionRatio
x64: CompressionRatio
x8: CompressionRatio

class AbortShardTransfer(_message.Message):
    __slots__ = ["from_peer_id", "shard_id", "to_peer_id", "to_shard_id"]
    FROM_PEER_ID_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    TO_PEER_ID_FIELD_NUMBER: _ClassVar[int]
    TO_SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    from_peer_id: int
    shard_id: int
    to_peer_id: int
    to_shard_id: int
    def __init__(self, shard_id: _Optional[int] = ..., to_shard_id: _Optional[int] = ..., from_peer_id: _Optional[int] = ..., to_peer_id: _Optional[int] = ...) -> None: ...

class AliasDescription(_message.Message):
    __slots__ = ["alias_name", "collection_name"]
    ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    alias_name: str
    collection_name: str
    def __init__(self, alias_name: _Optional[str] = ..., collection_name: _Optional[str] = ...) -> None: ...

class AliasOperations(_message.Message):
    __slots__ = ["create_alias", "delete_alias", "rename_alias"]
    CREATE_ALIAS_FIELD_NUMBER: _ClassVar[int]
    DELETE_ALIAS_FIELD_NUMBER: _ClassVar[int]
    RENAME_ALIAS_FIELD_NUMBER: _ClassVar[int]
    create_alias: CreateAlias
    delete_alias: DeleteAlias
    rename_alias: RenameAlias
    def __init__(self, create_alias: _Optional[_Union[CreateAlias, _Mapping]] = ..., rename_alias: _Optional[_Union[RenameAlias, _Mapping]] = ..., delete_alias: _Optional[_Union[DeleteAlias, _Mapping]] = ...) -> None: ...

class BinaryQuantization(_message.Message):
    __slots__ = ["always_ram"]
    ALWAYS_RAM_FIELD_NUMBER: _ClassVar[int]
    always_ram: bool
    def __init__(self, always_ram: bool = ...) -> None: ...

class ChangeAliases(_message.Message):
    __slots__ = ["actions", "timeout"]
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    actions: _containers.RepeatedCompositeFieldContainer[AliasOperations]
    timeout: int
    def __init__(self, actions: _Optional[_Iterable[_Union[AliasOperations, _Mapping]]] = ..., timeout: _Optional[int] = ...) -> None: ...

class CollectionClusterInfoRequest(_message.Message):
    __slots__ = ["collection_name"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    def __init__(self, collection_name: _Optional[str] = ...) -> None: ...

class CollectionClusterInfoResponse(_message.Message):
    __slots__ = ["local_shards", "peer_id", "remote_shards", "shard_count", "shard_transfers"]
    LOCAL_SHARDS_FIELD_NUMBER: _ClassVar[int]
    PEER_ID_FIELD_NUMBER: _ClassVar[int]
    REMOTE_SHARDS_FIELD_NUMBER: _ClassVar[int]
    SHARD_COUNT_FIELD_NUMBER: _ClassVar[int]
    SHARD_TRANSFERS_FIELD_NUMBER: _ClassVar[int]
    local_shards: _containers.RepeatedCompositeFieldContainer[LocalShardInfo]
    peer_id: int
    remote_shards: _containers.RepeatedCompositeFieldContainer[RemoteShardInfo]
    shard_count: int
    shard_transfers: _containers.RepeatedCompositeFieldContainer[ShardTransferInfo]
    def __init__(self, peer_id: _Optional[int] = ..., shard_count: _Optional[int] = ..., local_shards: _Optional[_Iterable[_Union[LocalShardInfo, _Mapping]]] = ..., remote_shards: _Optional[_Iterable[_Union[RemoteShardInfo, _Mapping]]] = ..., shard_transfers: _Optional[_Iterable[_Union[ShardTransferInfo, _Mapping]]] = ...) -> None: ...

class CollectionConfig(_message.Message):
    __slots__ = ["hnsw_config", "optimizer_config", "params", "quantization_config", "wal_config"]
    HNSW_CONFIG_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_CONFIG_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    WAL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    hnsw_config: HnswConfigDiff
    optimizer_config: OptimizersConfigDiff
    params: CollectionParams
    quantization_config: QuantizationConfig
    wal_config: WalConfigDiff
    def __init__(self, params: _Optional[_Union[CollectionParams, _Mapping]] = ..., hnsw_config: _Optional[_Union[HnswConfigDiff, _Mapping]] = ..., optimizer_config: _Optional[_Union[OptimizersConfigDiff, _Mapping]] = ..., wal_config: _Optional[_Union[WalConfigDiff, _Mapping]] = ..., quantization_config: _Optional[_Union[QuantizationConfig, _Mapping]] = ...) -> None: ...

class CollectionDescription(_message.Message):
    __slots__ = ["name"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    name: str
    def __init__(self, name: _Optional[str] = ...) -> None: ...

class CollectionExists(_message.Message):
    __slots__ = ["exists"]
    EXISTS_FIELD_NUMBER: _ClassVar[int]
    exists: bool
    def __init__(self, exists: bool = ...) -> None: ...

class CollectionExistsRequest(_message.Message):
    __slots__ = ["collection_name"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    def __init__(self, collection_name: _Optional[str] = ...) -> None: ...

class CollectionExistsResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: CollectionExists
    time: float
    def __init__(self, result: _Optional[_Union[CollectionExists, _Mapping]] = ..., time: _Optional[float] = ...) -> None: ...

class CollectionInfo(_message.Message):
    __slots__ = ["config", "indexed_vectors_count", "optimizer_status", "payload_schema", "points_count", "segments_count", "status", "vectors_count"]
    class PayloadSchemaEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: PayloadSchemaInfo
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[PayloadSchemaInfo, _Mapping]] = ...) -> None: ...
    CONFIG_FIELD_NUMBER: _ClassVar[int]
    INDEXED_VECTORS_COUNT_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZER_STATUS_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_SCHEMA_FIELD_NUMBER: _ClassVar[int]
    POINTS_COUNT_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_COUNT_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    VECTORS_COUNT_FIELD_NUMBER: _ClassVar[int]
    config: CollectionConfig
    indexed_vectors_count: int
    optimizer_status: OptimizerStatus
    payload_schema: _containers.MessageMap[str, PayloadSchemaInfo]
    points_count: int
    segments_count: int
    status: CollectionStatus
    vectors_count: int
    def __init__(self, status: _Optional[_Union[CollectionStatus, str]] = ..., optimizer_status: _Optional[_Union[OptimizerStatus, _Mapping]] = ..., vectors_count: _Optional[int] = ..., segments_count: _Optional[int] = ..., config: _Optional[_Union[CollectionConfig, _Mapping]] = ..., payload_schema: _Optional[_Mapping[str, PayloadSchemaInfo]] = ..., points_count: _Optional[int] = ..., indexed_vectors_count: _Optional[int] = ...) -> None: ...

class CollectionOperationResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: bool
    time: float
    def __init__(self, result: bool = ..., time: _Optional[float] = ...) -> None: ...

class CollectionParams(_message.Message):
    __slots__ = ["on_disk_payload", "read_fan_out_factor", "replication_factor", "shard_number", "sharding_method", "sparse_vectors_config", "vectors_config", "write_consistency_factor"]
    ON_DISK_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    READ_FAN_OUT_FACTOR_FIELD_NUMBER: _ClassVar[int]
    REPLICATION_FACTOR_FIELD_NUMBER: _ClassVar[int]
    SHARDING_METHOD_FIELD_NUMBER: _ClassVar[int]
    SHARD_NUMBER_FIELD_NUMBER: _ClassVar[int]
    SPARSE_VECTORS_CONFIG_FIELD_NUMBER: _ClassVar[int]
    VECTORS_CONFIG_FIELD_NUMBER: _ClassVar[int]
    WRITE_CONSISTENCY_FACTOR_FIELD_NUMBER: _ClassVar[int]
    on_disk_payload: bool
    read_fan_out_factor: int
    replication_factor: int
    shard_number: int
    sharding_method: ShardingMethod
    sparse_vectors_config: SparseVectorConfig
    vectors_config: VectorsConfig
    write_consistency_factor: int
    def __init__(self, shard_number: _Optional[int] = ..., on_disk_payload: bool = ..., vectors_config: _Optional[_Union[VectorsConfig, _Mapping]] = ..., replication_factor: _Optional[int] = ..., write_consistency_factor: _Optional[int] = ..., read_fan_out_factor: _Optional[int] = ..., sharding_method: _Optional[_Union[ShardingMethod, str]] = ..., sparse_vectors_config: _Optional[_Union[SparseVectorConfig, _Mapping]] = ...) -> None: ...

class CollectionParamsDiff(_message.Message):
    __slots__ = ["on_disk_payload", "read_fan_out_factor", "replication_factor", "write_consistency_factor"]
    ON_DISK_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    READ_FAN_OUT_FACTOR_FIELD_NUMBER: _ClassVar[int]
    REPLICATION_FACTOR_FIELD_NUMBER: _ClassVar[int]
    WRITE_CONSISTENCY_FACTOR_FIELD_NUMBER: _ClassVar[int]
    on_disk_payload: bool
    read_fan_out_factor: int
    replication_factor: int
    write_consistency_factor: int
    def __init__(self, replication_factor: _Optional[int] = ..., write_consistency_factor: _Optional[int] = ..., on_disk_payload: bool = ..., read_fan_out_factor: _Optional[int] = ...) -> None: ...

class CreateAlias(_message.Message):
    __slots__ = ["alias_name", "collection_name"]
    ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    alias_name: str
    collection_name: str
    def __init__(self, collection_name: _Optional[str] = ..., alias_name: _Optional[str] = ...) -> None: ...

class CreateCollection(_message.Message):
    __slots__ = ["collection_name", "hnsw_config", "init_from_collection", "on_disk_payload", "optimizers_config", "quantization_config", "replication_factor", "shard_number", "sharding_method", "sparse_vectors_config", "timeout", "vectors_config", "wal_config", "write_consistency_factor"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    HNSW_CONFIG_FIELD_NUMBER: _ClassVar[int]
    INIT_FROM_COLLECTION_FIELD_NUMBER: _ClassVar[int]
    ON_DISK_PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZERS_CONFIG_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    REPLICATION_FACTOR_FIELD_NUMBER: _ClassVar[int]
    SHARDING_METHOD_FIELD_NUMBER: _ClassVar[int]
    SHARD_NUMBER_FIELD_NUMBER: _ClassVar[int]
    SPARSE_VECTORS_CONFIG_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    VECTORS_CONFIG_FIELD_NUMBER: _ClassVar[int]
    WAL_CONFIG_FIELD_NUMBER: _ClassVar[int]
    WRITE_CONSISTENCY_FACTOR_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    hnsw_config: HnswConfigDiff
    init_from_collection: str
    on_disk_payload: bool
    optimizers_config: OptimizersConfigDiff
    quantization_config: QuantizationConfig
    replication_factor: int
    shard_number: int
    sharding_method: ShardingMethod
    sparse_vectors_config: SparseVectorConfig
    timeout: int
    vectors_config: VectorsConfig
    wal_config: WalConfigDiff
    write_consistency_factor: int
    def __init__(self, collection_name: _Optional[str] = ..., hnsw_config: _Optional[_Union[HnswConfigDiff, _Mapping]] = ..., wal_config: _Optional[_Union[WalConfigDiff, _Mapping]] = ..., optimizers_config: _Optional[_Union[OptimizersConfigDiff, _Mapping]] = ..., shard_number: _Optional[int] = ..., on_disk_payload: bool = ..., timeout: _Optional[int] = ..., vectors_config: _Optional[_Union[VectorsConfig, _Mapping]] = ..., replication_factor: _Optional[int] = ..., write_consistency_factor: _Optional[int] = ..., init_from_collection: _Optional[str] = ..., quantization_config: _Optional[_Union[QuantizationConfig, _Mapping]] = ..., sharding_method: _Optional[_Union[ShardingMethod, str]] = ..., sparse_vectors_config: _Optional[_Union[SparseVectorConfig, _Mapping]] = ...) -> None: ...

class CreateShardKey(_message.Message):
    __slots__ = ["placement", "replication_factor", "shard_key", "shards_number"]
    PLACEMENT_FIELD_NUMBER: _ClassVar[int]
    REPLICATION_FACTOR_FIELD_NUMBER: _ClassVar[int]
    SHARDS_NUMBER_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_FIELD_NUMBER: _ClassVar[int]
    placement: _containers.RepeatedScalarFieldContainer[int]
    replication_factor: int
    shard_key: ShardKey
    shards_number: int
    def __init__(self, shard_key: _Optional[_Union[ShardKey, _Mapping]] = ..., shards_number: _Optional[int] = ..., replication_factor: _Optional[int] = ..., placement: _Optional[_Iterable[int]] = ...) -> None: ...

class CreateShardKeyRequest(_message.Message):
    __slots__ = ["collection_name", "request", "timeout"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    request: CreateShardKey
    timeout: int
    def __init__(self, collection_name: _Optional[str] = ..., request: _Optional[_Union[CreateShardKey, _Mapping]] = ..., timeout: _Optional[int] = ...) -> None: ...

class CreateShardKeyResponse(_message.Message):
    __slots__ = ["result"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bool
    def __init__(self, result: bool = ...) -> None: ...

class DeleteAlias(_message.Message):
    __slots__ = ["alias_name"]
    ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    alias_name: str
    def __init__(self, alias_name: _Optional[str] = ...) -> None: ...

class DeleteCollection(_message.Message):
    __slots__ = ["collection_name", "timeout"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    timeout: int
    def __init__(self, collection_name: _Optional[str] = ..., timeout: _Optional[int] = ...) -> None: ...

class DeleteShardKey(_message.Message):
    __slots__ = ["shard_key"]
    SHARD_KEY_FIELD_NUMBER: _ClassVar[int]
    shard_key: ShardKey
    def __init__(self, shard_key: _Optional[_Union[ShardKey, _Mapping]] = ...) -> None: ...

class DeleteShardKeyRequest(_message.Message):
    __slots__ = ["collection_name", "request", "timeout"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    REQUEST_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    request: DeleteShardKey
    timeout: int
    def __init__(self, collection_name: _Optional[str] = ..., request: _Optional[_Union[DeleteShardKey, _Mapping]] = ..., timeout: _Optional[int] = ...) -> None: ...

class DeleteShardKeyResponse(_message.Message):
    __slots__ = ["result"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bool
    def __init__(self, result: bool = ...) -> None: ...

class Disabled(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetCollectionInfoRequest(_message.Message):
    __slots__ = ["collection_name"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    def __init__(self, collection_name: _Optional[str] = ...) -> None: ...

class GetCollectionInfoResponse(_message.Message):
    __slots__ = ["result", "time"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    result: CollectionInfo
    time: float
    def __init__(self, result: _Optional[_Union[CollectionInfo, _Mapping]] = ..., time: _Optional[float] = ...) -> None: ...

class HnswConfigDiff(_message.Message):
    __slots__ = ["ef_construct", "full_scan_threshold", "m", "max_indexing_threads", "on_disk", "payload_m"]
    EF_CONSTRUCT_FIELD_NUMBER: _ClassVar[int]
    FULL_SCAN_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_INDEXING_THREADS_FIELD_NUMBER: _ClassVar[int]
    M_FIELD_NUMBER: _ClassVar[int]
    ON_DISK_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_M_FIELD_NUMBER: _ClassVar[int]
    ef_construct: int
    full_scan_threshold: int
    m: int
    max_indexing_threads: int
    on_disk: bool
    payload_m: int
    def __init__(self, m: _Optional[int] = ..., ef_construct: _Optional[int] = ..., full_scan_threshold: _Optional[int] = ..., max_indexing_threads: _Optional[int] = ..., on_disk: bool = ..., payload_m: _Optional[int] = ...) -> None: ...

class IntegerIndexParams(_message.Message):
    __slots__ = ["lookup", "range"]
    LOOKUP_FIELD_NUMBER: _ClassVar[int]
    RANGE_FIELD_NUMBER: _ClassVar[int]
    lookup: bool
    range: bool
    def __init__(self, lookup: bool = ..., range: bool = ...) -> None: ...

class ListAliasesRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ListAliasesResponse(_message.Message):
    __slots__ = ["aliases", "time"]
    ALIASES_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    aliases: _containers.RepeatedCompositeFieldContainer[AliasDescription]
    time: float
    def __init__(self, aliases: _Optional[_Iterable[_Union[AliasDescription, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class ListCollectionAliasesRequest(_message.Message):
    __slots__ = ["collection_name"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    def __init__(self, collection_name: _Optional[str] = ...) -> None: ...

class ListCollectionsRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ListCollectionsResponse(_message.Message):
    __slots__ = ["collections", "time"]
    COLLECTIONS_FIELD_NUMBER: _ClassVar[int]
    TIME_FIELD_NUMBER: _ClassVar[int]
    collections: _containers.RepeatedCompositeFieldContainer[CollectionDescription]
    time: float
    def __init__(self, collections: _Optional[_Iterable[_Union[CollectionDescription, _Mapping]]] = ..., time: _Optional[float] = ...) -> None: ...

class LocalShardInfo(_message.Message):
    __slots__ = ["points_count", "shard_id", "shard_key", "state"]
    POINTS_COUNT_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    points_count: int
    shard_id: int
    shard_key: ShardKey
    state: ReplicaState
    def __init__(self, shard_id: _Optional[int] = ..., points_count: _Optional[int] = ..., state: _Optional[_Union[ReplicaState, str]] = ..., shard_key: _Optional[_Union[ShardKey, _Mapping]] = ...) -> None: ...

class MoveShard(_message.Message):
    __slots__ = ["from_peer_id", "method", "shard_id", "to_peer_id", "to_shard_id"]
    FROM_PEER_ID_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    TO_PEER_ID_FIELD_NUMBER: _ClassVar[int]
    TO_SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    from_peer_id: int
    method: ShardTransferMethod
    shard_id: int
    to_peer_id: int
    to_shard_id: int
    def __init__(self, shard_id: _Optional[int] = ..., to_shard_id: _Optional[int] = ..., from_peer_id: _Optional[int] = ..., to_peer_id: _Optional[int] = ..., method: _Optional[_Union[ShardTransferMethod, str]] = ...) -> None: ...

class MultiVectorConfig(_message.Message):
    __slots__ = ["comparator"]
    COMPARATOR_FIELD_NUMBER: _ClassVar[int]
    comparator: MultiVectorComparator
    def __init__(self, comparator: _Optional[_Union[MultiVectorComparator, str]] = ...) -> None: ...

class OptimizerStatus(_message.Message):
    __slots__ = ["error", "ok"]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    OK_FIELD_NUMBER: _ClassVar[int]
    error: str
    ok: bool
    def __init__(self, ok: bool = ..., error: _Optional[str] = ...) -> None: ...

class OptimizersConfigDiff(_message.Message):
    __slots__ = ["default_segment_number", "deleted_threshold", "flush_interval_sec", "indexing_threshold", "max_optimization_threads", "max_segment_size", "memmap_threshold", "vacuum_min_vector_number"]
    DEFAULT_SEGMENT_NUMBER_FIELD_NUMBER: _ClassVar[int]
    DELETED_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    FLUSH_INTERVAL_SEC_FIELD_NUMBER: _ClassVar[int]
    INDEXING_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    MAX_OPTIMIZATION_THREADS_FIELD_NUMBER: _ClassVar[int]
    MAX_SEGMENT_SIZE_FIELD_NUMBER: _ClassVar[int]
    MEMMAP_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    VACUUM_MIN_VECTOR_NUMBER_FIELD_NUMBER: _ClassVar[int]
    default_segment_number: int
    deleted_threshold: float
    flush_interval_sec: int
    indexing_threshold: int
    max_optimization_threads: int
    max_segment_size: int
    memmap_threshold: int
    vacuum_min_vector_number: int
    def __init__(self, deleted_threshold: _Optional[float] = ..., vacuum_min_vector_number: _Optional[int] = ..., default_segment_number: _Optional[int] = ..., max_segment_size: _Optional[int] = ..., memmap_threshold: _Optional[int] = ..., indexing_threshold: _Optional[int] = ..., flush_interval_sec: _Optional[int] = ..., max_optimization_threads: _Optional[int] = ...) -> None: ...

class PayloadIndexParams(_message.Message):
    __slots__ = ["integer_index_params", "text_index_params"]
    INTEGER_INDEX_PARAMS_FIELD_NUMBER: _ClassVar[int]
    TEXT_INDEX_PARAMS_FIELD_NUMBER: _ClassVar[int]
    integer_index_params: IntegerIndexParams
    text_index_params: TextIndexParams
    def __init__(self, text_index_params: _Optional[_Union[TextIndexParams, _Mapping]] = ..., integer_index_params: _Optional[_Union[IntegerIndexParams, _Mapping]] = ...) -> None: ...

class PayloadSchemaInfo(_message.Message):
    __slots__ = ["data_type", "params", "points"]
    DATA_TYPE_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    POINTS_FIELD_NUMBER: _ClassVar[int]
    data_type: PayloadSchemaType
    params: PayloadIndexParams
    points: int
    def __init__(self, data_type: _Optional[_Union[PayloadSchemaType, str]] = ..., params: _Optional[_Union[PayloadIndexParams, _Mapping]] = ..., points: _Optional[int] = ...) -> None: ...

class ProductQuantization(_message.Message):
    __slots__ = ["always_ram", "compression"]
    ALWAYS_RAM_FIELD_NUMBER: _ClassVar[int]
    COMPRESSION_FIELD_NUMBER: _ClassVar[int]
    always_ram: bool
    compression: CompressionRatio
    def __init__(self, compression: _Optional[_Union[CompressionRatio, str]] = ..., always_ram: bool = ...) -> None: ...

class QuantizationConfig(_message.Message):
    __slots__ = ["binary", "product", "scalar"]
    BINARY_FIELD_NUMBER: _ClassVar[int]
    PRODUCT_FIELD_NUMBER: _ClassVar[int]
    SCALAR_FIELD_NUMBER: _ClassVar[int]
    binary: BinaryQuantization
    product: ProductQuantization
    scalar: ScalarQuantization
    def __init__(self, scalar: _Optional[_Union[ScalarQuantization, _Mapping]] = ..., product: _Optional[_Union[ProductQuantization, _Mapping]] = ..., binary: _Optional[_Union[BinaryQuantization, _Mapping]] = ...) -> None: ...

class QuantizationConfigDiff(_message.Message):
    __slots__ = ["binary", "disabled", "product", "scalar"]
    BINARY_FIELD_NUMBER: _ClassVar[int]
    DISABLED_FIELD_NUMBER: _ClassVar[int]
    PRODUCT_FIELD_NUMBER: _ClassVar[int]
    SCALAR_FIELD_NUMBER: _ClassVar[int]
    binary: BinaryQuantization
    disabled: Disabled
    product: ProductQuantization
    scalar: ScalarQuantization
    def __init__(self, scalar: _Optional[_Union[ScalarQuantization, _Mapping]] = ..., product: _Optional[_Union[ProductQuantization, _Mapping]] = ..., disabled: _Optional[_Union[Disabled, _Mapping]] = ..., binary: _Optional[_Union[BinaryQuantization, _Mapping]] = ...) -> None: ...

class RemoteShardInfo(_message.Message):
    __slots__ = ["peer_id", "shard_id", "shard_key", "state"]
    PEER_ID_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    SHARD_KEY_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    peer_id: int
    shard_id: int
    shard_key: ShardKey
    state: ReplicaState
    def __init__(self, shard_id: _Optional[int] = ..., peer_id: _Optional[int] = ..., state: _Optional[_Union[ReplicaState, str]] = ..., shard_key: _Optional[_Union[ShardKey, _Mapping]] = ...) -> None: ...

class RenameAlias(_message.Message):
    __slots__ = ["new_alias_name", "old_alias_name"]
    NEW_ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    OLD_ALIAS_NAME_FIELD_NUMBER: _ClassVar[int]
    new_alias_name: str
    old_alias_name: str
    def __init__(self, old_alias_name: _Optional[str] = ..., new_alias_name: _Optional[str] = ...) -> None: ...

class Replica(_message.Message):
    __slots__ = ["peer_id", "shard_id"]
    PEER_ID_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    peer_id: int
    shard_id: int
    def __init__(self, shard_id: _Optional[int] = ..., peer_id: _Optional[int] = ...) -> None: ...

class ReplicateShard(_message.Message):
    __slots__ = ["from_peer_id", "method", "shard_id", "to_peer_id", "to_shard_id"]
    FROM_PEER_ID_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    TO_PEER_ID_FIELD_NUMBER: _ClassVar[int]
    TO_SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    from_peer_id: int
    method: ShardTransferMethod
    shard_id: int
    to_peer_id: int
    to_shard_id: int
    def __init__(self, shard_id: _Optional[int] = ..., to_shard_id: _Optional[int] = ..., from_peer_id: _Optional[int] = ..., to_peer_id: _Optional[int] = ..., method: _Optional[_Union[ShardTransferMethod, str]] = ...) -> None: ...

class RestartTransfer(_message.Message):
    __slots__ = ["from_peer_id", "method", "shard_id", "to_peer_id", "to_shard_id"]
    FROM_PEER_ID_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    TO_PEER_ID_FIELD_NUMBER: _ClassVar[int]
    TO_SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    from_peer_id: int
    method: ShardTransferMethod
    shard_id: int
    to_peer_id: int
    to_shard_id: int
    def __init__(self, shard_id: _Optional[int] = ..., to_shard_id: _Optional[int] = ..., from_peer_id: _Optional[int] = ..., to_peer_id: _Optional[int] = ..., method: _Optional[_Union[ShardTransferMethod, str]] = ...) -> None: ...

class ScalarQuantization(_message.Message):
    __slots__ = ["always_ram", "quantile", "type"]
    ALWAYS_RAM_FIELD_NUMBER: _ClassVar[int]
    QUANTILE_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    always_ram: bool
    quantile: float
    type: QuantizationType
    def __init__(self, type: _Optional[_Union[QuantizationType, str]] = ..., quantile: _Optional[float] = ..., always_ram: bool = ...) -> None: ...

class ShardKey(_message.Message):
    __slots__ = ["keyword", "number"]
    KEYWORD_FIELD_NUMBER: _ClassVar[int]
    NUMBER_FIELD_NUMBER: _ClassVar[int]
    keyword: str
    number: int
    def __init__(self, keyword: _Optional[str] = ..., number: _Optional[int] = ...) -> None: ...

class ShardTransferInfo(_message.Message):
    __slots__ = ["shard_id", "sync", "to", "to_shard_id"]
    FROM_FIELD_NUMBER: _ClassVar[int]
    SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    SYNC_FIELD_NUMBER: _ClassVar[int]
    TO_FIELD_NUMBER: _ClassVar[int]
    TO_SHARD_ID_FIELD_NUMBER: _ClassVar[int]
    shard_id: int
    sync: bool
    to: int
    to_shard_id: int
    def __init__(self, shard_id: _Optional[int] = ..., to_shard_id: _Optional[int] = ..., to: _Optional[int] = ..., sync: bool = ..., **kwargs) -> None: ...

class SparseIndexConfig(_message.Message):
    __slots__ = ["datatype", "full_scan_threshold", "on_disk"]
    DATATYPE_FIELD_NUMBER: _ClassVar[int]
    FULL_SCAN_THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    ON_DISK_FIELD_NUMBER: _ClassVar[int]
    datatype: Datatype
    full_scan_threshold: int
    on_disk: bool
    def __init__(self, full_scan_threshold: _Optional[int] = ..., on_disk: bool = ..., datatype: _Optional[_Union[Datatype, str]] = ...) -> None: ...

class SparseVectorConfig(_message.Message):
    __slots__ = ["map"]
    class MapEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: SparseVectorParams
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[SparseVectorParams, _Mapping]] = ...) -> None: ...
    MAP_FIELD_NUMBER: _ClassVar[int]
    map: _containers.MessageMap[str, SparseVectorParams]
    def __init__(self, map: _Optional[_Mapping[str, SparseVectorParams]] = ...) -> None: ...

class SparseVectorParams(_message.Message):
    __slots__ = ["index", "modifier"]
    INDEX_FIELD_NUMBER: _ClassVar[int]
    MODIFIER_FIELD_NUMBER: _ClassVar[int]
    index: SparseIndexConfig
    modifier: Modifier
    def __init__(self, index: _Optional[_Union[SparseIndexConfig, _Mapping]] = ..., modifier: _Optional[_Union[Modifier, str]] = ...) -> None: ...

class TextIndexParams(_message.Message):
    __slots__ = ["lowercase", "max_token_len", "min_token_len", "tokenizer"]
    LOWERCASE_FIELD_NUMBER: _ClassVar[int]
    MAX_TOKEN_LEN_FIELD_NUMBER: _ClassVar[int]
    MIN_TOKEN_LEN_FIELD_NUMBER: _ClassVar[int]
    TOKENIZER_FIELD_NUMBER: _ClassVar[int]
    lowercase: bool
    max_token_len: int
    min_token_len: int
    tokenizer: TokenizerType
    def __init__(self, tokenizer: _Optional[_Union[TokenizerType, str]] = ..., lowercase: bool = ..., min_token_len: _Optional[int] = ..., max_token_len: _Optional[int] = ...) -> None: ...

class UpdateCollection(_message.Message):
    __slots__ = ["collection_name", "hnsw_config", "optimizers_config", "params", "quantization_config", "sparse_vectors_config", "timeout", "vectors_config"]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    HNSW_CONFIG_FIELD_NUMBER: _ClassVar[int]
    OPTIMIZERS_CONFIG_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    SPARSE_VECTORS_CONFIG_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    VECTORS_CONFIG_FIELD_NUMBER: _ClassVar[int]
    collection_name: str
    hnsw_config: HnswConfigDiff
    optimizers_config: OptimizersConfigDiff
    params: CollectionParamsDiff
    quantization_config: QuantizationConfigDiff
    sparse_vectors_config: SparseVectorConfig
    timeout: int
    vectors_config: VectorsConfigDiff
    def __init__(self, collection_name: _Optional[str] = ..., optimizers_config: _Optional[_Union[OptimizersConfigDiff, _Mapping]] = ..., timeout: _Optional[int] = ..., params: _Optional[_Union[CollectionParamsDiff, _Mapping]] = ..., hnsw_config: _Optional[_Union[HnswConfigDiff, _Mapping]] = ..., vectors_config: _Optional[_Union[VectorsConfigDiff, _Mapping]] = ..., quantization_config: _Optional[_Union[QuantizationConfigDiff, _Mapping]] = ..., sparse_vectors_config: _Optional[_Union[SparseVectorConfig, _Mapping]] = ...) -> None: ...

class UpdateCollectionClusterSetupRequest(_message.Message):
    __slots__ = ["abort_transfer", "collection_name", "create_shard_key", "delete_shard_key", "drop_replica", "move_shard", "replicate_shard", "restart_transfer", "timeout"]
    ABORT_TRANSFER_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_NAME_FIELD_NUMBER: _ClassVar[int]
    CREATE_SHARD_KEY_FIELD_NUMBER: _ClassVar[int]
    DELETE_SHARD_KEY_FIELD_NUMBER: _ClassVar[int]
    DROP_REPLICA_FIELD_NUMBER: _ClassVar[int]
    MOVE_SHARD_FIELD_NUMBER: _ClassVar[int]
    REPLICATE_SHARD_FIELD_NUMBER: _ClassVar[int]
    RESTART_TRANSFER_FIELD_NUMBER: _ClassVar[int]
    TIMEOUT_FIELD_NUMBER: _ClassVar[int]
    abort_transfer: AbortShardTransfer
    collection_name: str
    create_shard_key: CreateShardKey
    delete_shard_key: DeleteShardKey
    drop_replica: Replica
    move_shard: MoveShard
    replicate_shard: ReplicateShard
    restart_transfer: RestartTransfer
    timeout: int
    def __init__(self, collection_name: _Optional[str] = ..., move_shard: _Optional[_Union[MoveShard, _Mapping]] = ..., replicate_shard: _Optional[_Union[ReplicateShard, _Mapping]] = ..., abort_transfer: _Optional[_Union[AbortShardTransfer, _Mapping]] = ..., drop_replica: _Optional[_Union[Replica, _Mapping]] = ..., create_shard_key: _Optional[_Union[CreateShardKey, _Mapping]] = ..., delete_shard_key: _Optional[_Union[DeleteShardKey, _Mapping]] = ..., restart_transfer: _Optional[_Union[RestartTransfer, _Mapping]] = ..., timeout: _Optional[int] = ...) -> None: ...

class UpdateCollectionClusterSetupResponse(_message.Message):
    __slots__ = ["result"]
    RESULT_FIELD_NUMBER: _ClassVar[int]
    result: bool
    def __init__(self, result: bool = ...) -> None: ...

class VectorParams(_message.Message):
    __slots__ = ["datatype", "distance", "hnsw_config", "multivector_config", "on_disk", "quantization_config", "size"]
    DATATYPE_FIELD_NUMBER: _ClassVar[int]
    DISTANCE_FIELD_NUMBER: _ClassVar[int]
    HNSW_CONFIG_FIELD_NUMBER: _ClassVar[int]
    MULTIVECTOR_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ON_DISK_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    SIZE_FIELD_NUMBER: _ClassVar[int]
    datatype: Datatype
    distance: Distance
    hnsw_config: HnswConfigDiff
    multivector_config: MultiVectorConfig
    on_disk: bool
    quantization_config: QuantizationConfig
    size: int
    def __init__(self, size: _Optional[int] = ..., distance: _Optional[_Union[Distance, str]] = ..., hnsw_config: _Optional[_Union[HnswConfigDiff, _Mapping]] = ..., quantization_config: _Optional[_Union[QuantizationConfig, _Mapping]] = ..., on_disk: bool = ..., datatype: _Optional[_Union[Datatype, str]] = ..., multivector_config: _Optional[_Union[MultiVectorConfig, _Mapping]] = ...) -> None: ...

class VectorParamsDiff(_message.Message):
    __slots__ = ["hnsw_config", "on_disk", "quantization_config"]
    HNSW_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ON_DISK_FIELD_NUMBER: _ClassVar[int]
    QUANTIZATION_CONFIG_FIELD_NUMBER: _ClassVar[int]
    hnsw_config: HnswConfigDiff
    on_disk: bool
    quantization_config: QuantizationConfigDiff
    def __init__(self, hnsw_config: _Optional[_Union[HnswConfigDiff, _Mapping]] = ..., quantization_config: _Optional[_Union[QuantizationConfigDiff, _Mapping]] = ..., on_disk: bool = ...) -> None: ...

class VectorParamsDiffMap(_message.Message):
    __slots__ = ["map"]
    class MapEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: VectorParamsDiff
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[VectorParamsDiff, _Mapping]] = ...) -> None: ...
    MAP_FIELD_NUMBER: _ClassVar[int]
    map: _containers.MessageMap[str, VectorParamsDiff]
    def __init__(self, map: _Optional[_Mapping[str, VectorParamsDiff]] = ...) -> None: ...

class VectorParamsMap(_message.Message):
    __slots__ = ["map"]
    class MapEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: VectorParams
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[VectorParams, _Mapping]] = ...) -> None: ...
    MAP_FIELD_NUMBER: _ClassVar[int]
    map: _containers.MessageMap[str, VectorParams]
    def __init__(self, map: _Optional[_Mapping[str, VectorParams]] = ...) -> None: ...

class VectorsConfig(_message.Message):
    __slots__ = ["params", "params_map"]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    PARAMS_MAP_FIELD_NUMBER: _ClassVar[int]
    params: VectorParams
    params_map: VectorParamsMap
    def __init__(self, params: _Optional[_Union[VectorParams, _Mapping]] = ..., params_map: _Optional[_Union[VectorParamsMap, _Mapping]] = ...) -> None: ...

class VectorsConfigDiff(_message.Message):
    __slots__ = ["params", "params_map"]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    PARAMS_MAP_FIELD_NUMBER: _ClassVar[int]
    params: VectorParamsDiff
    params_map: VectorParamsDiffMap
    def __init__(self, params: _Optional[_Union[VectorParamsDiff, _Mapping]] = ..., params_map: _Optional[_Union[VectorParamsDiffMap, _Mapping]] = ...) -> None: ...

class WalConfigDiff(_message.Message):
    __slots__ = ["wal_capacity_mb", "wal_segments_ahead"]
    WAL_CAPACITY_MB_FIELD_NUMBER: _ClassVar[int]
    WAL_SEGMENTS_AHEAD_FIELD_NUMBER: _ClassVar[int]
    wal_capacity_mb: int
    wal_segments_ahead: int
    def __init__(self, wal_capacity_mb: _Optional[int] = ..., wal_segments_ahead: _Optional[int] = ...) -> None: ...

class Datatype(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class Modifier(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class MultiVectorComparator(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class Distance(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class CollectionStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class PayloadSchemaType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class QuantizationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class CompressionRatio(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class ShardingMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class TokenizerType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class ReplicaState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class ShardTransferMethod(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
