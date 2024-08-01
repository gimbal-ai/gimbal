from gogoproto import gogo_pb2 as _gogo_pb2
from google.protobuf import timestamp_pb2 as _timestamp_pb2
from src.api.corepb.v1 import cp_edge_pb2 as _cp_edge_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor
DEVICE_STATUS_DISCONNECTED: DeviceStatus
DEVICE_STATUS_HEALTHY: DeviceStatus
DEVICE_STATUS_UNHEALTHY: DeviceStatus
DEVICE_STATUS_UNKNOWN: DeviceStatus
OS_KIND_ANDROID: OSKind
OS_KIND_IOS: OSKind
OS_KIND_LINUX: OSKind
OS_KIND_MACOS: OSKind
OS_KIND_OTHER: OSKind
OS_KIND_UNKNOWN: OSKind
OS_KIND_WINDOWS: OSKind

class AssociateTagsWithDeployKeyRequest(_message.Message):
    __slots__ = ["deploy_key_id", "fleet_id", "tags"]
    class TagsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    DEPLOY_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    deploy_key_id: _uuid_pb2.UUID
    fleet_id: _uuid_pb2.UUID
    tags: _containers.ScalarMap[str, str]
    def __init__(self, deploy_key_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., tags: _Optional[_Mapping[str, str]] = ..., fleet_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class AssociateTagsWithDeployKeyResponse(_message.Message):
    __slots__ = ["deploy_key_id", "tags"]
    class TagsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tag
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...
    DEPLOY_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    deploy_key_id: _uuid_pb2.UUID
    tags: _containers.MessageMap[str, Tag]
    def __init__(self, deploy_key_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., tags: _Optional[_Mapping[str, Tag]] = ...) -> None: ...

class CreateFleetRequest(_message.Message):
    __slots__ = ["fleet"]
    FLEET_FIELD_NUMBER: _ClassVar[int]
    fleet: FleetInfo
    def __init__(self, fleet: _Optional[_Union[FleetInfo, _Mapping]] = ...) -> None: ...

class CreateFleetResponse(_message.Message):
    __slots__ = ["fleet"]
    FLEET_FIELD_NUMBER: _ClassVar[int]
    fleet: FleetInfo
    def __init__(self, fleet: _Optional[_Union[FleetInfo, _Mapping]] = ...) -> None: ...

class DeleteDefaultTagRequest(_message.Message):
    __slots__ = ["fleet_id", "key"]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    fleet_id: _uuid_pb2.UUID
    key: str
    def __init__(self, fleet_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., key: _Optional[str] = ...) -> None: ...

class DeleteDefaultTagResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class DeleteDevicesRequest(_message.Message):
    __slots__ = ["ids"]
    IDS_FIELD_NUMBER: _ClassVar[int]
    ids: _containers.RepeatedCompositeFieldContainer[_uuid_pb2.UUID]
    def __init__(self, ids: _Optional[_Iterable[_Union[_uuid_pb2.UUID, _Mapping]]] = ...) -> None: ...

class DeleteDevicesResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class DeleteTagRequest(_message.Message):
    __slots__ = ["device_id", "key"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    KEY_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    key: str
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., key: _Optional[str] = ...) -> None: ...

class DeleteTagResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class DeviceInfo(_message.Message):
    __slots__ = ["capabilities", "created_at", "display_name", "fleet_id", "hostname", "id", "last_heartbeat_ns", "os", "serial", "status", "tags", "version"]
    class TagsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tag
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_NAME_FIELD_NUMBER: _ClassVar[int]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    LAST_HEARTBEAT_NS_FIELD_NUMBER: _ClassVar[int]
    OS_FIELD_NUMBER: _ClassVar[int]
    SERIAL_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    capabilities: _cp_edge_pb2.DeviceCapabilities
    created_at: _timestamp_pb2.Timestamp
    display_name: str
    fleet_id: _uuid_pb2.UUID
    hostname: str
    id: _uuid_pb2.UUID
    last_heartbeat_ns: int
    os: OSInfo
    serial: str
    status: DeviceStatus
    tags: _containers.MessageMap[str, Tag]
    version: str
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., serial: _Optional[str] = ..., hostname: _Optional[str] = ..., display_name: _Optional[str] = ..., fleet_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., last_heartbeat_ns: _Optional[int] = ..., status: _Optional[_Union[DeviceStatus, str]] = ..., capabilities: _Optional[_Union[_cp_edge_pb2.DeviceCapabilities, _Mapping]] = ..., tags: _Optional[_Mapping[str, Tag]] = ..., os: _Optional[_Union[OSInfo, _Mapping]] = ..., version: _Optional[str] = ...) -> None: ...

class FleetInfo(_message.Message):
    __slots__ = ["created_at", "description", "id", "name", "org_id", "tags"]
    class TagsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tag
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ORG_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    created_at: _timestamp_pb2.Timestamp
    description: str
    id: _uuid_pb2.UUID
    name: str
    org_id: _uuid_pb2.UUID
    tags: _containers.MessageMap[str, Tag]
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., org_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., tags: _Optional[_Mapping[str, Tag]] = ...) -> None: ...

class GetDefaultTagsRequest(_message.Message):
    __slots__ = ["fleet_id"]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    fleet_id: _uuid_pb2.UUID
    def __init__(self, fleet_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetDefaultTagsResponse(_message.Message):
    __slots__ = ["tags"]
    class TagsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tag
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...
    TAGS_FIELD_NUMBER: _ClassVar[int]
    tags: _containers.MessageMap[str, Tag]
    def __init__(self, tags: _Optional[_Mapping[str, Tag]] = ...) -> None: ...

class GetDeviceRequest(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetDeviceResponse(_message.Message):
    __slots__ = ["device"]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    device: DeviceInfo
    def __init__(self, device: _Optional[_Union[DeviceInfo, _Mapping]] = ...) -> None: ...

class GetFleetByNameRequest(_message.Message):
    __slots__ = ["name", "org_id"]
    NAME_FIELD_NUMBER: _ClassVar[int]
    ORG_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    org_id: _uuid_pb2.UUID
    def __init__(self, name: _Optional[str] = ..., org_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetFleetByNameResponse(_message.Message):
    __slots__ = ["fleet"]
    FLEET_FIELD_NUMBER: _ClassVar[int]
    fleet: FleetInfo
    def __init__(self, fleet: _Optional[_Union[FleetInfo, _Mapping]] = ...) -> None: ...

class GetFleetRequest(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetFleetResponse(_message.Message):
    __slots__ = ["fleet"]
    FLEET_FIELD_NUMBER: _ClassVar[int]
    fleet: FleetInfo
    def __init__(self, fleet: _Optional[_Union[FleetInfo, _Mapping]] = ...) -> None: ...

class GetTagsRequest(_message.Message):
    __slots__ = ["device_id"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetTagsResponse(_message.Message):
    __slots__ = ["tags"]
    class TagsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tag
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...
    TAGS_FIELD_NUMBER: _ClassVar[int]
    tags: _containers.MessageMap[str, Tag]
    def __init__(self, tags: _Optional[_Mapping[str, Tag]] = ...) -> None: ...

class ListDevicesRequest(_message.Message):
    __slots__ = ["fleet_id"]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    fleet_id: _uuid_pb2.UUID
    def __init__(self, fleet_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class ListDevicesResponse(_message.Message):
    __slots__ = ["devices"]
    DEVICES_FIELD_NUMBER: _ClassVar[int]
    devices: _containers.RepeatedCompositeFieldContainer[DeviceInfo]
    def __init__(self, devices: _Optional[_Iterable[_Union[DeviceInfo, _Mapping]]] = ...) -> None: ...

class ListFleetsRequest(_message.Message):
    __slots__ = ["org_id"]
    ORG_ID_FIELD_NUMBER: _ClassVar[int]
    org_id: _uuid_pb2.UUID
    def __init__(self, org_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class ListFleetsResponse(_message.Message):
    __slots__ = ["fleets"]
    FLEETS_FIELD_NUMBER: _ClassVar[int]
    fleets: _containers.RepeatedCompositeFieldContainer[FleetInfo]
    def __init__(self, fleets: _Optional[_Iterable[_Union[FleetInfo, _Mapping]]] = ...) -> None: ...

class ListTagsAssociatedWithDeployKeyRequest(_message.Message):
    __slots__ = ["deploy_key_id"]
    DEPLOY_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    deploy_key_id: _uuid_pb2.UUID
    def __init__(self, deploy_key_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class ListTagsAssociatedWithDeployKeyResponse(_message.Message):
    __slots__ = ["deploy_key_id", "tags"]
    class TagsEntry(_message.Message):
        __slots__ = ["key", "value"]
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Tag
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...
    DEPLOY_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    TAGS_FIELD_NUMBER: _ClassVar[int]
    deploy_key_id: _uuid_pb2.UUID
    tags: _containers.MessageMap[str, Tag]
    def __init__(self, deploy_key_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., tags: _Optional[_Mapping[str, Tag]] = ...) -> None: ...

class OSInfo(_message.Message):
    __slots__ = ["kind", "version"]
    KIND_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    kind: OSKind
    version: str
    def __init__(self, kind: _Optional[_Union[OSKind, str]] = ..., version: _Optional[str] = ...) -> None: ...

class RegisterRequest(_message.Message):
    __slots__ = ["device_serial", "hostname", "os", "version"]
    DEVICE_SERIAL_FIELD_NUMBER: _ClassVar[int]
    HOSTNAME_FIELD_NUMBER: _ClassVar[int]
    OS_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    device_serial: str
    hostname: str
    os: OSInfo
    version: str
    def __init__(self, device_serial: _Optional[str] = ..., hostname: _Optional[str] = ..., os: _Optional[_Union[OSInfo, _Mapping]] = ..., version: _Optional[str] = ...) -> None: ...

class RegisterResponse(_message.Message):
    __slots__ = ["device_id"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class SetDeviceCapabilitiesRequest(_message.Message):
    __slots__ = ["capabilities", "device_id"]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    capabilities: _cp_edge_pb2.DeviceCapabilities
    device_id: _uuid_pb2.UUID
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., capabilities: _Optional[_Union[_cp_edge_pb2.DeviceCapabilities, _Mapping]] = ...) -> None: ...

class SetDeviceCapabilitiesResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class Tag(_message.Message):
    __slots__ = ["key", "metadata", "value"]
    KEY_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    key: str
    metadata: TagMetadata
    value: str
    def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ..., metadata: _Optional[_Union[TagMetadata, _Mapping]] = ...) -> None: ...

class TagMetadata(_message.Message):
    __slots__ = ["created_at", "is_inherited", "updated_at"]
    CREATED_AT_FIELD_NUMBER: _ClassVar[int]
    IS_INHERITED_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_FIELD_NUMBER: _ClassVar[int]
    created_at: _timestamp_pb2.Timestamp
    is_inherited: bool
    updated_at: _timestamp_pb2.Timestamp
    def __init__(self, is_inherited: bool = ..., updated_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ..., created_at: _Optional[_Union[_timestamp_pb2.Timestamp, _Mapping]] = ...) -> None: ...

class UnassociateTagsWithDeployKeyRequest(_message.Message):
    __slots__ = ["deploy_key_id"]
    DEPLOY_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    deploy_key_id: _uuid_pb2.UUID
    def __init__(self, deploy_key_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class UnassociateTagsWithDeployKeyResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class UpdateDeviceRequest(_message.Message):
    __slots__ = ["deleted_tag_keys", "device"]
    DELETED_TAG_KEYS_FIELD_NUMBER: _ClassVar[int]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    deleted_tag_keys: _containers.RepeatedScalarFieldContainer[str]
    device: DeviceInfo
    def __init__(self, device: _Optional[_Union[DeviceInfo, _Mapping]] = ..., deleted_tag_keys: _Optional[_Iterable[str]] = ...) -> None: ...

class UpdateDeviceResponse(_message.Message):
    __slots__ = ["device"]
    DEVICE_FIELD_NUMBER: _ClassVar[int]
    device: DeviceInfo
    def __init__(self, device: _Optional[_Union[DeviceInfo, _Mapping]] = ...) -> None: ...

class UpdateFleetRequest(_message.Message):
    __slots__ = ["deleted_tag_keys", "fleet"]
    DELETED_TAG_KEYS_FIELD_NUMBER: _ClassVar[int]
    FLEET_FIELD_NUMBER: _ClassVar[int]
    deleted_tag_keys: _containers.RepeatedScalarFieldContainer[str]
    fleet: FleetInfo
    def __init__(self, fleet: _Optional[_Union[FleetInfo, _Mapping]] = ..., deleted_tag_keys: _Optional[_Iterable[str]] = ...) -> None: ...

class UpdateFleetResponse(_message.Message):
    __slots__ = ["fleet"]
    FLEET_FIELD_NUMBER: _ClassVar[int]
    fleet: FleetInfo
    def __init__(self, fleet: _Optional[_Union[FleetInfo, _Mapping]] = ...) -> None: ...

class UpdateStatusRequest(_message.Message):
    __slots__ = ["device_id"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class UpdateStatusResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class UpsertDefaultTagRequest(_message.Message):
    __slots__ = ["fleet_id", "tag"]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    fleet_id: _uuid_pb2.UUID
    tag: Tag
    def __init__(self, fleet_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., tag: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...

class UpsertDefaultTagResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class UpsertTagRequest(_message.Message):
    __slots__ = ["device_id", "tag"]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    TAG_FIELD_NUMBER: _ClassVar[int]
    device_id: _uuid_pb2.UUID
    tag: Tag
    def __init__(self, device_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., tag: _Optional[_Union[Tag, _Mapping]] = ...) -> None: ...

class UpsertTagResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class DeviceStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []

class OSKind(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = []
