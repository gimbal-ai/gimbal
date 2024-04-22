from gogoproto import gogo_pb2 as _gogo_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DeviceJWTClaims(_message.Message):
    __slots__ = ["deploy_key_id", "device_id", "fleet_id"]
    DEPLOY_KEY_ID_FIELD_NUMBER: _ClassVar[int]
    DEVICE_ID_FIELD_NUMBER: _ClassVar[int]
    FLEET_ID_FIELD_NUMBER: _ClassVar[int]
    deploy_key_id: str
    device_id: str
    fleet_id: str
    def __init__(self, device_id: _Optional[str] = ..., fleet_id: _Optional[str] = ..., deploy_key_id: _Optional[str] = ...) -> None: ...

class JWTClaims(_message.Message):
    __slots__ = ["audience", "device_claims", "expires_at", "issued_at", "issuer", "jti", "not_before", "scopes", "service_claims", "subject", "user_claims"]
    AUDIENCE_FIELD_NUMBER: _ClassVar[int]
    DEVICE_CLAIMS_FIELD_NUMBER: _ClassVar[int]
    EXPIRES_AT_FIELD_NUMBER: _ClassVar[int]
    ISSUED_AT_FIELD_NUMBER: _ClassVar[int]
    ISSUER_FIELD_NUMBER: _ClassVar[int]
    JTI_FIELD_NUMBER: _ClassVar[int]
    NOT_BEFORE_FIELD_NUMBER: _ClassVar[int]
    SCOPES_FIELD_NUMBER: _ClassVar[int]
    SERVICE_CLAIMS_FIELD_NUMBER: _ClassVar[int]
    SUBJECT_FIELD_NUMBER: _ClassVar[int]
    USER_CLAIMS_FIELD_NUMBER: _ClassVar[int]
    audience: str
    device_claims: DeviceJWTClaims
    expires_at: int
    issued_at: int
    issuer: str
    jti: str
    not_before: int
    scopes: _containers.RepeatedScalarFieldContainer[str]
    service_claims: ServiceJWTClaims
    subject: str
    user_claims: UserJWTClaims
    def __init__(self, audience: _Optional[str] = ..., expires_at: _Optional[int] = ..., jti: _Optional[str] = ..., issued_at: _Optional[int] = ..., issuer: _Optional[str] = ..., not_before: _Optional[int] = ..., subject: _Optional[str] = ..., scopes: _Optional[_Iterable[str]] = ..., user_claims: _Optional[_Union[UserJWTClaims, _Mapping]] = ..., service_claims: _Optional[_Union[ServiceJWTClaims, _Mapping]] = ..., device_claims: _Optional[_Union[DeviceJWTClaims, _Mapping]] = ...) -> None: ...

class ServiceJWTClaims(_message.Message):
    __slots__ = ["service_id"]
    SERVICE_ID_FIELD_NUMBER: _ClassVar[int]
    service_id: str
    def __init__(self, service_id: _Optional[str] = ...) -> None: ...

class UserJWTClaims(_message.Message):
    __slots__ = ["authorizations", "email", "user_id"]
    class AuthorizationDetails(_message.Message):
        __slots__ = ["org_ids", "scopes"]
        ORG_IDS_FIELD_NUMBER: _ClassVar[int]
        SCOPES_FIELD_NUMBER: _ClassVar[int]
        org_ids: _containers.RepeatedScalarFieldContainer[str]
        scopes: _containers.RepeatedScalarFieldContainer[str]
        def __init__(self, scopes: _Optional[_Iterable[str]] = ..., org_ids: _Optional[_Iterable[str]] = ...) -> None: ...
    AUTHORIZATIONS_FIELD_NUMBER: _ClassVar[int]
    EMAIL_FIELD_NUMBER: _ClassVar[int]
    USER_ID_FIELD_NUMBER: _ClassVar[int]
    authorizations: _containers.RepeatedCompositeFieldContainer[UserJWTClaims.AuthorizationDetails]
    email: str
    user_id: str
    def __init__(self, user_id: _Optional[str] = ..., email: _Optional[str] = ..., authorizations: _Optional[_Iterable[_Union[UserJWTClaims.AuthorizationDetails, _Mapping]]] = ...) -> None: ...
