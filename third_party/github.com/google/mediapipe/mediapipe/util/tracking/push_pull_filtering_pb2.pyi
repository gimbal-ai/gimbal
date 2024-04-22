from google.protobuf.internal import python_message as _python_message
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class PushPullOptions(_message.Message):
    __slots__ = ["bilateral_sigma", "pull_bilateral_scale", "pull_propagation_scale", "push_bilateral_scale", "push_propagation_scale"]
    BILATERAL_SIGMA_FIELD_NUMBER: _ClassVar[int]
    Extensions: _python_message._ExtensionDict
    PULL_BILATERAL_SCALE_FIELD_NUMBER: _ClassVar[int]
    PULL_PROPAGATION_SCALE_FIELD_NUMBER: _ClassVar[int]
    PUSH_BILATERAL_SCALE_FIELD_NUMBER: _ClassVar[int]
    PUSH_PROPAGATION_SCALE_FIELD_NUMBER: _ClassVar[int]
    bilateral_sigma: float
    pull_bilateral_scale: float
    pull_propagation_scale: float
    push_bilateral_scale: float
    push_propagation_scale: float
    def __init__(self, bilateral_sigma: _Optional[float] = ..., pull_propagation_scale: _Optional[float] = ..., push_propagation_scale: _Optional[float] = ..., pull_bilateral_scale: _Optional[float] = ..., push_bilateral_scale: _Optional[float] = ...) -> None: ...
