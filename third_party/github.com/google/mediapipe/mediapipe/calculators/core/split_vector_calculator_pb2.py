# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/core/split_vector_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n8mediapipe/calculators/core/split_vector_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"/\n\x05Range\x12\x14\n\x05\x62\x65gin\x18\x01 \x01(\x05R\x05\x62\x65gin\x12\x10\n\x03\x65nd\x18\x02 \x01(\x05R\x03\x65nd\"\xfe\x01\n\x1cSplitVectorCalculatorOptions\x12(\n\x06ranges\x18\x01 \x03(\x0b\x32\x10.mediapipe.RangeR\x06ranges\x12(\n\x0c\x65lement_only\x18\x02 \x01(\x08:\x05\x66\x61lseR\x0b\x65lementOnly\x12.\n\x0f\x63ombine_outputs\x18\x03 \x01(\x08:\x05\x66\x61lseR\x0e\x63ombineOutputs2Z\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x8e\xed\xda{ \x01(\x0b\x32\'.mediapipe.SplitVectorCalculatorOptionsR\x03\x65xtB8Z6github.com/google/mediapipe/mediapipe/calculators/core')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.core.split_vector_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_SPLITVECTORCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z6github.com/google/mediapipe/mediapipe/calculators/core'
  _RANGE._serialized_start=109
  _RANGE._serialized_end=156
  _SPLITVECTORCALCULATOROPTIONS._serialized_start=159
  _SPLITVECTORCALCULATOROPTIONS._serialized_end=413
# @@protoc_insertion_point(module_scope)
