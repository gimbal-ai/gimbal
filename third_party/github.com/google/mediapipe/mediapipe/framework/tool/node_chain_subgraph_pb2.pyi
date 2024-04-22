from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class NodeChainSubgraphOptions(_message.Message):
    __slots__ = ["chain_length", "node_type"]
    CHAIN_LENGTH_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    NODE_TYPE_FIELD_NUMBER: _ClassVar[int]
    chain_length: int
    ext: _descriptor.FieldDescriptor
    node_type: str
    def __init__(self, node_type: _Optional[str] = ..., chain_length: _Optional[int] = ...) -> None: ...
