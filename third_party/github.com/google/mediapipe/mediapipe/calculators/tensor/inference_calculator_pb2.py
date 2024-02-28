# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/tensor/inference_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2
from mediapipe.framework import calculator_options_pb2 as mediapipe_dot_framework_dot_calculator__options__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n7mediapipe/calculators/tensor/inference_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a,mediapipe/framework/calculator_options.proto\"\xdb\x0c\n\x1aInferenceCalculatorOptions\x12\x1d\n\nmodel_path\x18\x01 \x01(\tR\tmodelPath\x12\"\n\x07use_gpu\x18\x02 \x01(\x08:\x05\x66\x61lseB\x02\x18\x01R\x06useGpu\x12&\n\tuse_nnapi\x18\x03 \x01(\x08:\x05\x66\x61lseB\x02\x18\x01R\x08useNnapi\x12(\n\x0e\x63pu_num_thread\x18\x04 \x01(\x05:\x02-1R\x0c\x63puNumThread\x12J\n\x08\x64\x65legate\x18\x05 \x01(\x0b\x32..mediapipe.InferenceCalculatorOptions.DelegateR\x08\x64\x65legate\x1a\x80\n\n\x08\x44\x65legate\x12O\n\x06tflite\x18\x01 \x01(\x0b\x32\x35.mediapipe.InferenceCalculatorOptions.Delegate.TfLiteH\x00R\x06tflite\x12\x46\n\x03gpu\x18\x02 \x01(\x0b\x32\x32.mediapipe.InferenceCalculatorOptions.Delegate.GpuH\x00R\x03gpu\x12L\n\x05nnapi\x18\x03 \x01(\x0b\x32\x34.mediapipe.InferenceCalculatorOptions.Delegate.NnapiH\x00R\x05nnapi\x12R\n\x07xnnpack\x18\x04 \x01(\x0b\x32\x36.mediapipe.InferenceCalculatorOptions.Delegate.XnnpackH\x00R\x07xnnpack\x1a\x08\n\x06TfLite\x1a\x80\x06\n\x03Gpu\x12\x36\n\x14use_advanced_gpu_api\x18\x01 \x01(\x08:\x05\x66\x61lseR\x11useAdvancedGpuApi\x12M\n\x03\x61pi\x18\x04 \x01(\x0e\x32\x36.mediapipe.InferenceCalculatorOptions.Delegate.Gpu.Api:\x03\x41NYR\x03\x61pi\x12\x36\n\x14\x61llow_precision_loss\x18\x03 \x01(\x08:\x04trueR\x12\x61llowPrecisionLoss\x12,\n\x12\x63\x61\x63hed_kernel_path\x18\x02 \x01(\tR\x10\x63\x61\x63hedKernelPath\x12\x30\n\x14serialized_model_dir\x18\x07 \x01(\tR\x12serializedModelDir\x12\x8d\x01\n\x16\x63\x61\x63he_writing_behavior\x18\n \x01(\x0e\x32G.mediapipe.InferenceCalculatorOptions.Delegate.Gpu.CacheWritingBehavior:\x0eWRITE_OR_ERRORR\x14\x63\x61\x63heWritingBehavior\x12\x1f\n\x0bmodel_token\x18\x08 \x01(\tR\nmodelToken\x12h\n\x05usage\x18\x05 \x01(\x0e\x32\x41.mediapipe.InferenceCalculatorOptions.Delegate.Gpu.InferenceUsage:\x0fSUSTAINED_SPEEDR\x05usage\"&\n\x03\x41pi\x12\x07\n\x03\x41NY\x10\x00\x12\n\n\x06OPENGL\x10\x01\x12\n\n\x06OPENCL\x10\x02\"G\n\x14\x43\x61\x63heWritingBehavior\x12\x0c\n\x08NO_WRITE\x10\x00\x12\r\n\tTRY_WRITE\x10\x01\x12\x12\n\x0eWRITE_OR_ERROR\x10\x02\"N\n\x0eInferenceUsage\x12\x0f\n\x0bUNSPECIFIED\x10\x00\x12\x16\n\x12\x46\x41ST_SINGLE_ANSWER\x10\x01\x12\x13\n\x0fSUSTAINED_SPEED\x10\x02\x1ap\n\x05Nnapi\x12\x1b\n\tcache_dir\x18\x01 \x01(\tR\x08\x63\x61\x63heDir\x12\x1f\n\x0bmodel_token\x18\x02 \x01(\tR\nmodelToken\x12)\n\x10\x61\x63\x63\x65lerator_name\x18\x03 \x01(\tR\x0f\x61\x63\x63\x65leratorName\x1a.\n\x07Xnnpack\x12#\n\x0bnum_threads\x18\x01 \x01(\x05:\x02-1R\nnumThreadsB\n\n\x08\x64\x65legate2Y\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xf7\xd3\xcb\xa0\x01 \x01(\x0b\x32%.mediapipe.InferenceCalculatorOptionsR\x03\x65xtB{\n%com.google.mediapipe.calculator.protoB\x18InferenceCalculatorProtoZ8github.com/google/mediapipe/mediapipe/calculators/tensor')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.tensor.inference_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_INFERENCECALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n%com.google.mediapipe.calculator.protoB\030InferenceCalculatorProtoZ8github.com/google/mediapipe/mediapipe/calculators/tensor'
  _INFERENCECALCULATOROPTIONS.fields_by_name['use_gpu']._options = None
  _INFERENCECALCULATOROPTIONS.fields_by_name['use_gpu']._serialized_options = b'\030\001'
  _INFERENCECALCULATOROPTIONS.fields_by_name['use_nnapi']._options = None
  _INFERENCECALCULATOROPTIONS.fields_by_name['use_nnapi']._serialized_options = b'\030\001'
  _INFERENCECALCULATOROPTIONS._serialized_start=155
  _INFERENCECALCULATOROPTIONS._serialized_end=1782
  _INFERENCECALCULATOROPTIONS_DELEGATE._serialized_start=411
  _INFERENCECALCULATOROPTIONS_DELEGATE._serialized_end=1691
  _INFERENCECALCULATOROPTIONS_DELEGATE_TFLITE._serialized_start=738
  _INFERENCECALCULATOROPTIONS_DELEGATE_TFLITE._serialized_end=746
  _INFERENCECALCULATOROPTIONS_DELEGATE_GPU._serialized_start=749
  _INFERENCECALCULATOROPTIONS_DELEGATE_GPU._serialized_end=1517
  _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_API._serialized_start=1326
  _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_API._serialized_end=1364
  _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_CACHEWRITINGBEHAVIOR._serialized_start=1366
  _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_CACHEWRITINGBEHAVIOR._serialized_end=1437
  _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_INFERENCEUSAGE._serialized_start=1439
  _INFERENCECALCULATOROPTIONS_DELEGATE_GPU_INFERENCEUSAGE._serialized_end=1517
  _INFERENCECALCULATOROPTIONS_DELEGATE_NNAPI._serialized_start=1519
  _INFERENCECALCULATOROPTIONS_DELEGATE_NNAPI._serialized_end=1631
  _INFERENCECALCULATOROPTIONS_DELEGATE_XNNPACK._serialized_start=1633
  _INFERENCECALCULATOROPTIONS_DELEGATE_XNNPACK._serialized_end=1679
# @@protoc_insertion_point(module_scope)
