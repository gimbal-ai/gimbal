# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/gpu/gl_scaler_calculator.proto
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
from mediapipe.gpu import scale_mode_pb2 as mediapipe_dot_gpu_dot_scale__mode__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n(mediapipe/gpu/gl_scaler_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a\x1emediapipe/gpu/scale_mode.proto\"\xda\x03\n\x19GlScalerCalculatorOptions\x12!\n\x0coutput_width\x18\x01 \x01(\x05R\x0boutputWidth\x12#\n\routput_height\x18\x02 \x01(\x05R\x0coutputHeight\x12$\n\x0coutput_scale\x18\x07 \x01(\x02:\x01\x31R\x0boutputScale\x12\x1a\n\x08rotation\x18\x03 \x01(\x05R\x08rotation\x12#\n\rflip_vertical\x18\x04 \x01(\x08R\x0c\x66lipVertical\x12\'\n\x0f\x66lip_horizontal\x18\x05 \x01(\x08R\x0e\x66lipHorizontal\x12\x38\n\nscale_mode\x18\x06 \x01(\x0e\x32\x19.mediapipe.ScaleMode.ModeR\tscaleMode\x12R\n\"use_nearest_neighbor_interpolation\x18\x08 \x01(\x08:\x05\x66\x61lseR\x1fuseNearestNeighborInterpolation2W\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\x96\xcd\xaaO \x01(\x0b\x32$.mediapipe.GlScalerCalculatorOptionsR\x03\x65xtB+Z)github.com/google/mediapipe/mediapipe/gpu')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.gpu.gl_scaler_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_GLSCALERCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z)github.com/google/mediapipe/mediapipe/gpu'
  _GLSCALERCALCULATOROPTIONS._serialized_start=126
  _GLSCALERCALCULATOROPTIONS._serialized_end=600
# @@protoc_insertion_point(module_scope)