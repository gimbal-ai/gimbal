# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/video/flow_to_image_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n:mediapipe/calculators/video/flow_to_image_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xbd\x01\n\x1c\x46lowToImageCalculatorOptions\x12 \n\tmin_value\x18\x01 \x01(\x02:\x03-40R\x08minValue\x12\x1f\n\tmax_value\x18\x02 \x01(\x02:\x02\x34\x30R\x08maxValue2Z\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xf0\xbb\x92! \x01(\x0b\x32\'.mediapipe.FlowToImageCalculatorOptionsR\x03\x65xtB9Z7github.com/google/mediapipe/mediapipe/calculators/video')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.video.flow_to_image_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_FLOWTOIMAGECALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z7github.com/google/mediapipe/mediapipe/calculators/video'
  _FLOWTOIMAGECALCULATOROPTIONS._serialized_start=112
  _FLOWTOIMAGECALCULATOROPTIONS._serialized_end=301
# @@protoc_insertion_point(module_scope)
