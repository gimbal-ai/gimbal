from gogoproto import gogo_pb2 as _gogo_pb2
from google.protobuf import wrappers_pb2 as _wrappers_pb2
from src.common.typespb import uuid_pb2 as _uuid_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CreateOrgRequest(_message.Message):
    __slots__ = ["creator_id", "org_info"]
    CREATOR_ID_FIELD_NUMBER: _ClassVar[int]
    ORG_INFO_FIELD_NUMBER: _ClassVar[int]
    creator_id: _uuid_pb2.UUID
    org_info: OrgInfo
    def __init__(self, org_info: _Optional[_Union[OrgInfo, _Mapping]] = ..., creator_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class CreateOrgResponse(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class DeleteOrgRequest(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class DeleteOrgResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class DeleteUserRequest(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class DeleteUserResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GetOrgRequest(_message.Message):
    __slots__ = ["id", "org_name"]
    ID_FIELD_NUMBER: _ClassVar[int]
    ORG_NAME_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    org_name: str
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., org_name: _Optional[str] = ...) -> None: ...

class GetOrgResponse(_message.Message):
    __slots__ = ["org_info"]
    ORG_INFO_FIELD_NUMBER: _ClassVar[int]
    org_info: OrgInfo
    def __init__(self, org_info: _Optional[_Union[OrgInfo, _Mapping]] = ...) -> None: ...

class GetUserRequest(_message.Message):
    __slots__ = ["id"]
    ID_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetUserResponse(_message.Message):
    __slots__ = ["user_info"]
    USER_INFO_FIELD_NUMBER: _ClassVar[int]
    user_info: UserInfo
    def __init__(self, user_info: _Optional[_Union[UserInfo, _Mapping]] = ...) -> None: ...

class GetUsersRequest(_message.Message):
    __slots__ = ["org_id"]
    ORG_ID_FIELD_NUMBER: _ClassVar[int]
    org_id: _uuid_pb2.UUID
    def __init__(self, org_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class GetUsersResponse(_message.Message):
    __slots__ = ["user_roles"]
    USER_ROLES_FIELD_NUMBER: _ClassVar[int]
    user_roles: _containers.RepeatedCompositeFieldContainer[UserRoleInfo]
    def __init__(self, user_roles: _Optional[_Iterable[_Union[UserRoleInfo, _Mapping]]] = ...) -> None: ...

class GrantUserScopesRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class GrantUserScopesResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class ListOrgsRequest(_message.Message):
    __slots__ = ["user_id"]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    user_id: _uuid_pb2.UUID
    def __init__(self, user_id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ...) -> None: ...

class ListOrgsResponse(_message.Message):
    __slots__ = ["org_ids"]
    ORG_IDS_FIELD_NUMBER: _ClassVar[int]
    org_ids: _containers.RepeatedCompositeFieldContainer[_uuid_pb2.UUID]
    def __init__(self, org_ids: _Optional[_Iterable[_Union[_uuid_pb2.UUID, _Mapping]]] = ...) -> None: ...

class OrgInfo(_message.Message):
    __slots__ = ["id", "org_name"]
    ID_FIELD_NUMBER: _ClassVar[int]
    ORG_NAME_FIELD_NUMBER: _ClassVar[int]
    id: _uuid_pb2.UUID
    org_name: str
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., org_name: _Optional[str] = ...) -> None: ...

class RevokeUserScopesRequest(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class RevokeUserScopesResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class UpdateUserRequest(_message.Message):
    __slots__ = ["display_picture", "id"]
    DISPLAY_PICTURE_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    display_picture: _wrappers_pb2.StringValue
    id: _uuid_pb2.UUID
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., display_picture: _Optional[_Union[_wrappers_pb2.StringValue, _Mapping]] = ...) -> None: ...

class UpdateUserResponse(_message.Message):
    __slots__ = []
    def __init__(self) -> None: ...

class UpsertUserRequest(_message.Message):
    __slots__ = ["user_info"]
    USER_INFO_FIELD_NUMBER: _ClassVar[int]
    user_info: UserInfo
    def __init__(self, user_info: _Optional[_Union[UserInfo, _Mapping]] = ...) -> None: ...

class UpsertUserResponse(_message.Message):
    __slots__ = ["created", "user_info"]
    CREATED_FIELD_NUMBER: _ClassVar[int]
    USER_INFO_FIELD_NUMBER: _ClassVar[int]
    created: bool
    user_info: UserInfo
    def __init__(self, user_info: _Optional[_Union[UserInfo, _Mapping]] = ..., created: bool = ...) -> None: ...

class UserInfo(_message.Message):
    __slots__ = ["auth_provider_id", "display_picture", "email", "id", "identity_provider", "name"]
    AUTH_PROVIDER_ID_FIELD_NUMBER: _ClassVar[int]
    DISPLAY_PICTURE_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_PROVIDER_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    auth_provider_id: str
    display_picture: str
    email: str
    id: _uuid_pb2.UUID
    identity_provider: str
    name: str
    def __init__(self, id: _Optional[_Union[_uuid_pb2.UUID, _Mapping]] = ..., name: _Optional[str] = ..., email: _Optional[str] = ..., display_picture: _Optional[str] = ..., identity_provider: _Optional[str] = ..., auth_provider_id: _Optional[str] = ...) -> None: ...

class UserRoleInfo(_message.Message):
    __slots__ = ["role_name", "user_info"]
    ROLE_NAME_FIELD_NUMBER: _ClassVar[int]
    USER_INFO_FIELD_NUMBER: _ClassVar[int]
    role_name: str
    user_info: UserInfo
    def __init__(self, user_info: _Optional[_Union[UserInfo, _Mapping]] = ..., role_name: _Optional[str] = ...) -> None: ...
