from mediapipe.framework import mediapipe_options_pb2 as _mediapipe_options_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ThreadPoolExecutorOptions(_message.Message):
    __slots__ = ["nice_priority_level", "num_threads", "require_processor_performance", "stack_size", "thread_name_prefix"]
    class ProcessorPerformance(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = []
    EXT_FIELD_NUMBER: _ClassVar[int]
    NICE_PRIORITY_LEVEL_FIELD_NUMBER: _ClassVar[int]
    NUM_THREADS_FIELD_NUMBER: _ClassVar[int]
    PROCESSOR_PERFORMANCE_HIGH: ThreadPoolExecutorOptions.ProcessorPerformance
    PROCESSOR_PERFORMANCE_LOW: ThreadPoolExecutorOptions.ProcessorPerformance
    PROCESSOR_PERFORMANCE_NORMAL: ThreadPoolExecutorOptions.ProcessorPerformance
    REQUIRE_PROCESSOR_PERFORMANCE_FIELD_NUMBER: _ClassVar[int]
    STACK_SIZE_FIELD_NUMBER: _ClassVar[int]
    THREAD_NAME_PREFIX_FIELD_NUMBER: _ClassVar[int]
    ext: _descriptor.FieldDescriptor
    nice_priority_level: int
    num_threads: int
    require_processor_performance: ThreadPoolExecutorOptions.ProcessorPerformance
    stack_size: int
    thread_name_prefix: str
    def __init__(self, num_threads: _Optional[int] = ..., stack_size: _Optional[int] = ..., nice_priority_level: _Optional[int] = ..., require_processor_performance: _Optional[_Union[ThreadPoolExecutorOptions.ProcessorPerformance, str]] = ..., thread_name_prefix: _Optional[str] = ...) -> None: ...
