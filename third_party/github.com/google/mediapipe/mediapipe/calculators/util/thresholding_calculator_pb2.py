# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/thresholding_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n8mediapipe/calculators/util/thresholding_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\x9a\x01\n\x1dThresholdingCalculatorOptions\x12\x1c\n\tthreshold\x18\x01 \x01(\x01R\tthreshold2[\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xe2\xc7\xfc{ \x01(\x0b\x32(.mediapipe.ThresholdingCalculatorOptionsR\x03\x65xtB8Z6github.com/google/mediapipe/mediapipe/calculators/util')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.thresholding_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_THRESHOLDINGCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z6github.com/google/mediapipe/mediapipe/calculators/util'
  _THRESHOLDINGCALCULATOROPTIONS._serialized_start=110
  _THRESHOLDINGCALCULATOROPTIONS._serialized_end=264
# @@protoc_insertion_point(module_scope)
