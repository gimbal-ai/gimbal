# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/flat_color_image_calculator.proto
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
from mediapipe.util import color_pb2 as mediapipe_dot_util_dot_color__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n<mediapipe/calculators/util/flat_color_image_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a\x1amediapipe/util/color.proto\"\xf1\x01\n\x1f\x46latColorImageCalculatorOptions\x12!\n\x0coutput_width\x18\x01 \x01(\x05R\x0boutputWidth\x12#\n\routput_height\x18\x02 \x01(\x05R\x0coutputHeight\x12&\n\x05\x63olor\x18\x03 \x01(\x0b\x32\x10.mediapipe.ColorR\x05\x63olor2^\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x93\xca\xea\xf5\x01 \x01(\x0b\x32*.mediapipe.FlatColorImageCalculatorOptionsR\x03\x65xtB8Z6github.com/google/mediapipe/mediapipe/calculators/util')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.flat_color_image_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_FLATCOLORIMAGECALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z6github.com/google/mediapipe/mediapipe/calculators/util'
  _FLATCOLORIMAGECALCULATOROPTIONS._serialized_start=142
  _FLATCOLORIMAGECALCULATOROPTIONS._serialized_end=383
# @@protoc_insertion_point(module_scope)
