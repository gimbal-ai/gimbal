# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/tensor/tensors_to_landmarks_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nBmediapipe/calculators/tensor/tensors_to_landmarks_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xc9\x05\n#TensorsToLandmarksCalculatorOptions\x12#\n\rnum_landmarks\x18\x01 \x01(\x05R\x0cnumLandmarks\x12*\n\x11input_image_width\x18\x02 \x01(\x05R\x0finputImageWidth\x12,\n\x12input_image_height\x18\x03 \x01(\x05R\x10inputImageHeight\x12.\n\x0f\x66lip_vertically\x18\x04 \x01(\x08:\x05\x66\x61lseR\x0e\x66lipVertically\x12\x32\n\x11\x66lip_horizontally\x18\x06 \x01(\x08:\x05\x66\x61lseR\x10\x66lipHorizontally\x12\"\n\x0bnormalize_z\x18\x05 \x01(\x02:\x01\x31R\nnormalizeZ\x12\x7f\n\x15visibility_activation\x18\x07 \x01(\x0e\x32\x39.mediapipe.TensorsToLandmarksCalculatorOptions.Activation:\x0f\x41\x43TIVATION_NONER\x14visibilityActivation\x12{\n\x13presence_activation\x18\x08 \x01(\x0e\x32\x39.mediapipe.TensorsToLandmarksCalculatorOptions.Activation:\x0f\x41\x43TIVATION_NONER\x12presenceActivation\"9\n\nActivation\x12\x13\n\x0f\x41\x43TIVATION_NONE\x10\x00\x12\x16\n\x12\x41\x43TIVATION_SIGMOID\x10\x01\x32\x62\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xb0\x8d\x8c\xa0\x01 \x01(\x0b\x32..mediapipe.TensorsToLandmarksCalculatorOptionsR\x03\x65xtB:Z8github.com/google/mediapipe/mediapipe/calculators/tensor')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.tensor.tensors_to_landmarks_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_TENSORSTOLANDMARKSCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z8github.com/google/mediapipe/mediapipe/calculators/tensor'
  _TENSORSTOLANDMARKSCALCULATOROPTIONS._serialized_start=120
  _TENSORSTOLANDMARKSCALCULATOROPTIONS._serialized_end=833
  _TENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION._serialized_start=676
  _TENSORSTOLANDMARKSCALCULATOROPTIONS_ACTIVATION._serialized_end=733
# @@protoc_insertion_point(module_scope)
