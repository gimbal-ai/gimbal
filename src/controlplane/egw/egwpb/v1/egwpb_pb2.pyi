from google.protobuf import any_pb2 as _any_pb2
from src.api.corepb.v1 import cp_edge_pb2 as _cp_edge_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class BridgeRequest(_message.Message):
    __slots__ = ["msg", "topic"]
    MSG_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    msg: _any_pb2.Any
    topic: _cp_edge_pb2.EdgeCPTopic
    def __init__(self, topic: _Optional[_Union[_cp_edge_pb2.EdgeCPTopic, str]] = ..., msg: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...

class BridgeResponse(_message.Message):
    __slots__ = ["msg", "topic"]
    MSG_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    msg: _any_pb2.Any
    topic: _cp_edge_pb2.CPEdgeTopic
    def __init__(self, topic: _Optional[_Union[_cp_edge_pb2.CPEdgeTopic, str]] = ..., msg: _Optional[_Union[_any_pb2.Any, _Mapping]] = ...) -> None: ...
