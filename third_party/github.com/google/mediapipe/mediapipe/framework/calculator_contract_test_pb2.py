# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/framework/calculator_contract_test.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n2mediapipe/framework/calculator_contract_test.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\x9f\x01\n\x1d\x43\x61lculatorContractTestOptions\x12!\n\ntest_field\x18\x01 \x01(\x01:\x02-1R\ttestField2[\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xb7\xd5\x80Z \x01(\x0b\x32(.mediapipe.CalculatorContractTestOptionsR\x03\x65xtB1Z/github.com/google/mediapipe/mediapipe/framework')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.framework.calculator_contract_test_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_CALCULATORCONTRACTTESTOPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z/github.com/google/mediapipe/mediapipe/framework'
  _CALCULATORCONTRACTTESTOPTIONS._serialized_start=104
  _CALCULATORCONTRACTTESTOPTIONS._serialized_end=263
# @@protoc_insertion_point(module_scope)
