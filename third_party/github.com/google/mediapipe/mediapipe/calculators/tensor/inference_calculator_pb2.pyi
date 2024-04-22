from mediapipe.framework import calculator_pb2 as _calculator_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2
from mediapipe.framework import calculator_options_pb2 as _calculator_options_pb2_1
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InferenceCalculatorOptions(_message.Message):
    __slots__ = ["cpu_num_thread", "delegate", "model_path", "use_gpu", "use_nnapi"]
    class Delegate(_message.Message):
        __slots__ = ["gpu", "nnapi", "tflite", "xnnpack"]
        class Gpu(_message.Message):
            __slots__ = ["allow_precision_loss", "api", "cache_writing_behavior", "cached_kernel_path", "model_token", "serialized_model_dir", "usage", "use_advanced_gpu_api"]
            class Api(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
                __slots__ = []
            class CacheWritingBehavior(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
                __slots__ = []
            class InferenceUsage(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
                __slots__ = []
            ALLOW_PRECISION_LOSS_FIELD_NUMBER: _ClassVar[int]
            ANY: InferenceCalculatorOptions.Delegate.Gpu.Api
            API_FIELD_NUMBER: _ClassVar[int]
            CACHED_KERNEL_PATH_FIELD_NUMBER: _ClassVar[int]
            CACHE_WRITING_BEHAVIOR_FIELD_NUMBER: _ClassVar[int]
            FAST_SINGLE_ANSWER: InferenceCalculatorOptions.Delegate.Gpu.InferenceUsage
            MODEL_TOKEN_FIELD_NUMBER: _ClassVar[int]
            NO_WRITE: InferenceCalculatorOptions.Delegate.Gpu.CacheWritingBehavior
            OPENCL: InferenceCalculatorOptions.Delegate.Gpu.Api
            OPENGL: InferenceCalculatorOptions.Delegate.Gpu.Api
            SERIALIZED_MODEL_DIR_FIELD_NUMBER: _ClassVar[int]
            SUSTAINED_SPEED: InferenceCalculatorOptions.Delegate.Gpu.InferenceUsage
            TRY_WRITE: InferenceCalculatorOptions.Delegate.Gpu.CacheWritingBehavior
            UNSPECIFIED: InferenceCalculatorOptions.Delegate.Gpu.InferenceUsage
            USAGE_FIELD_NUMBER: _ClassVar[int]
            USE_ADVANCED_GPU_API_FIELD_NUMBER: _ClassVar[int]
            WRITE_OR_ERROR: InferenceCalculatorOptions.Delegate.Gpu.CacheWritingBehavior
            allow_precision_loss: bool
            api: InferenceCalculatorOptions.Delegate.Gpu.Api
            cache_writing_behavior: InferenceCalculatorOptions.Delegate.Gpu.CacheWritingBehavior
            cached_kernel_path: str
            model_token: str
            serialized_model_dir: str
            usage: InferenceCalculatorOptions.Delegate.Gpu.InferenceUsage
            use_advanced_gpu_api: bool
            def __init__(self, use_advanced_gpu_api: bool = ..., api: _Optional[_Union[InferenceCalculatorOptions.Delegate.Gpu.Api, str]] = ..., allow_precision_loss: bool = ..., cached_kernel_path: _Optional[str] = ..., serialized_model_dir: _Optional[str] = ..., cache_writing_behavior: _Optional[_Union[InferenceCalculatorOptions.Delegate.Gpu.CacheWritingBehavior, str]] = ..., model_token: _Optional[str] = ..., usage: _Optional[_Union[InferenceCalculatorOptions.Delegate.Gpu.InferenceUsage, str]] = ...) -> None: ...
        class Nnapi(_message.Message):
            __slots__ = ["accelerator_name", "cache_dir", "model_token"]
            ACCELERATOR_NAME_FIELD_NUMBER: _ClassVar[int]
            CACHE_DIR_FIELD_NUMBER: _ClassVar[int]
            MODEL_TOKEN_FIELD_NUMBER: _ClassVar[int]
            accelerator_name: str
            cache_dir: str
            model_token: str
            def __init__(self, cache_dir: _Optional[str] = ..., model_token: _Optional[str] = ..., accelerator_name: _Optional[str] = ...) -> None: ...
        class TfLite(_message.Message):
            __slots__ = []
            def __init__(self) -> None: ...
        class Xnnpack(_message.Message):
            __slots__ = ["num_threads"]
            NUM_THREADS_FIELD_NUMBER: _ClassVar[int]
            num_threads: int
            def __init__(self, num_threads: _Optional[int] = ...) -> None: ...
        GPU_FIELD_NUMBER: _ClassVar[int]
        NNAPI_FIELD_NUMBER: _ClassVar[int]
        TFLITE_FIELD_NUMBER: _ClassVar[int]
        XNNPACK_FIELD_NUMBER: _ClassVar[int]
        gpu: InferenceCalculatorOptions.Delegate.Gpu
        nnapi: InferenceCalculatorOptions.Delegate.Nnapi
        tflite: InferenceCalculatorOptions.Delegate.TfLite
        xnnpack: InferenceCalculatorOptions.Delegate.Xnnpack
        def __init__(self, tflite: _Optional[_Union[InferenceCalculatorOptions.Delegate.TfLite, _Mapping]] = ..., gpu: _Optional[_Union[InferenceCalculatorOptions.Delegate.Gpu, _Mapping]] = ..., nnapi: _Optional[_Union[InferenceCalculatorOptions.Delegate.Nnapi, _Mapping]] = ..., xnnpack: _Optional[_Union[InferenceCalculatorOptions.Delegate.Xnnpack, _Mapping]] = ...) -> None: ...
    CPU_NUM_THREAD_FIELD_NUMBER: _ClassVar[int]
    DELEGATE_FIELD_NUMBER: _ClassVar[int]
    EXT_FIELD_NUMBER: _ClassVar[int]
    MODEL_PATH_FIELD_NUMBER: _ClassVar[int]
    USE_GPU_FIELD_NUMBER: _ClassVar[int]
    USE_NNAPI_FIELD_NUMBER: _ClassVar[int]
    cpu_num_thread: int
    delegate: InferenceCalculatorOptions.Delegate
    ext: _descriptor.FieldDescriptor
    model_path: str
    use_gpu: bool
    use_nnapi: bool
    def __init__(self, model_path: _Optional[str] = ..., use_gpu: bool = ..., use_nnapi: bool = ..., cpu_num_thread: _Optional[int] = ..., delegate: _Optional[_Union[InferenceCalculatorOptions.Delegate, _Mapping]] = ...) -> None: ...
