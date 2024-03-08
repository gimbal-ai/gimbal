# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/image/recolor_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n4mediapipe/calculators/image/recolor_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a\x1amediapipe/util/color.proto\"\xb9\x03\n\x18RecolorCalculatorOptions\x12\x64\n\x0cmask_channel\x18\x01 \x01(\x0e\x32/.mediapipe.RecolorCalculatorOptions.MaskChannel:\x10MASK_CHANNEL_REDR\x0bmaskChannel\x12&\n\x05\x63olor\x18\x02 \x01(\x0b\x32\x10.mediapipe.ColorR\x05\x63olor\x12&\n\x0binvert_mask\x18\x03 \x01(\x08:\x05\x66\x61lseR\ninvertMask\x12\x38\n\x15\x61\x64just_with_luminance\x18\x04 \x01(\x08:\x04trueR\x13\x61\x64justWithLuminance\"U\n\x0bMaskChannel\x12\x18\n\x14MASK_CHANNEL_UNKNOWN\x10\x00\x12\x14\n\x10MASK_CHANNEL_RED\x10\x01\x12\x16\n\x12MASK_CHANNEL_ALPHA\x10\x02\x32V\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x8d\x84\xb5x \x01(\x0b\x32#.mediapipe.RecolorCalculatorOptionsR\x03\x65xtB9Z7github.com/google/mediapipe/mediapipe/calculators/image')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.image.recolor_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_RECOLORCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z7github.com/google/mediapipe/mediapipe/calculators/image'
  _RECOLORCALCULATOROPTIONS._serialized_start=134
  _RECOLORCALCULATOROPTIONS._serialized_end=575
  _RECOLORCALCULATOROPTIONS_MASKCHANNEL._serialized_start=402
  _RECOLORCALCULATOROPTIONS_MASKCHANNEL._serialized_end=487
# @@protoc_insertion_point(module_scope)