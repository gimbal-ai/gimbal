from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class QdrantSearchCalculatorOptions(_message.Message):
    __slots__ = ["address", "collection", "limit", "payload_key"]
    ADDRESS_FIELD_NUMBER: _ClassVar[int]
    COLLECTION_FIELD_NUMBER: _ClassVar[int]
    LIMIT_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_KEY_FIELD_NUMBER: _ClassVar[int]
    address: str
    collection: str
    limit: int
    payload_key: str
    def __init__(self, address: _Optional[str] = ..., collection: _Optional[str] = ..., limit: _Optional[int] = ..., payload_key: _Optional[str] = ...) -> None: ...
