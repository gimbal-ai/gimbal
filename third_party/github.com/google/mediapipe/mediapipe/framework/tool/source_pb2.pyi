from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SidePacketsToStreamsCalculatorOptions(_message.Message):
    __slots__ = ["num_inputs", "set_timestamp", "vectors_of_packets"]
    class SetTimestampMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    EXT_FIELD_NUMBER: _ClassVar[int]
    NUM_INPUTS_FIELD_NUMBER: _ClassVar[int]
    SET_TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_MODE_NONE: SidePacketsToStreamsCalculatorOptions.SetTimestampMode
    TIMESTAMP_MODE_PRE_STREAM: SidePacketsToStreamsCalculatorOptions.SetTimestampMode
    TIMESTAMP_MODE_VECTOR_INDEX: SidePacketsToStreamsCalculatorOptions.SetTimestampMode
    TIMESTAMP_MODE_WHOLE_STREAM: SidePacketsToStreamsCalculatorOptions.SetTimestampMode
    VECTORS_OF_PACKETS_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    num_inputs: int
    set_timestamp: SidePacketsToStreamsCalculatorOptions.SetTimestampMode
    vectors_of_packets: bool
    def __init__(self, num_inputs: _Optional[int] = ..., set_timestamp: _Optional[_Union[SidePacketsToStreamsCalculatorOptions.SetTimestampMode, str]] = ..., vectors_of_packets: bool = ...) -> None: ...
