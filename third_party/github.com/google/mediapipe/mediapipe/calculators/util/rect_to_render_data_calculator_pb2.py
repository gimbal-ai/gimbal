# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/rect_to_render_data_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n?mediapipe/calculators/util/rect_to_render_data_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a\x1amediapipe/util/color.proto\"\xae\x02\n!RectToRenderDataCalculatorOptions\x12\x16\n\x06\x66illed\x18\x01 \x01(\x08R\x06\x66illed\x12&\n\x05\x63olor\x18\x02 \x01(\x0b\x32\x10.mediapipe.ColorR\x05\x63olor\x12\x1f\n\tthickness\x18\x03 \x01(\x01:\x01\x31R\tthickness\x12\x19\n\x04oval\x18\x04 \x01(\x08:\x05\x66\x61lseR\x04oval\x12,\n\x12top_left_thickness\x18\x05 \x01(\x01R\x10topLeftThickness2_\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xac\xdb\x87} \x01(\x0b\x32,.mediapipe.RectToRenderDataCalculatorOptionsR\x03\x65xtB8Z6github.com/google/mediapipe/mediapipe/calculators/util')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.rect_to_render_data_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_RECTTORENDERDATACALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z6github.com/google/mediapipe/mediapipe/calculators/util'
  _RECTTORENDERDATACALCULATOROPTIONS._serialized_start=145
  _RECTTORENDERDATACALCULATOROPTIONS._serialized_end=447
# @@protoc_insertion_point(module_scope)
