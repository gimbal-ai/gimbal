# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/gpu/gl_context_options.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n&mediapipe/gpu/gl_context_options.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\x8a\x01\n\x10GlContextOptions\x12&\n\x0fgl_context_name\x18\x01 \x01(\tR\rglContextName2N\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x82\x89\x82j \x01(\x0b\x32\x1b.mediapipe.GlContextOptionsR\x03\x65xtB+Z)github.com/google/mediapipe/mediapipe/gpu')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.gpu.gl_context_options_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_GLCONTEXTOPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z)github.com/google/mediapipe/mediapipe/gpu'
  _GLCONTEXTOPTIONS._serialized_start=92
  _GLCONTEXTOPTIONS._serialized_end=230
# @@protoc_insertion_point(module_scope)
