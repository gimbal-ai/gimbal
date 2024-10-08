# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/tensor/tensors_to_audio_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n>mediapipe/calculators/tensor/tensors_to_audio_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xa2\x04\n\x1fTensorsToAudioCalculatorOptions\x12\x19\n\x08\x66\x66t_size\x18\x01 \x01(\x03R\x07\x66\x66tSize\x12\x1f\n\x0bnum_samples\x18\x02 \x01(\x03R\nnumSamples\x12\x39\n\x17num_overlapping_samples\x18\x03 \x01(\x03:\x01\x30R\x15numOverlappingSamples\x12x\n\x11\x64\x66t_tensor_format\x18\x0b \x01(\x0e\x32:.mediapipe.TensorsToAudioCalculatorOptions.DftTensorFormat:\x10T2A_WITH_NYQUISTR\x0f\x64\x66tTensorFormat\x12$\n\x0evolume_gain_db\x18\x0c \x01(\x01R\x0cvolumeGainDb\"\x87\x01\n\x0f\x44\x66tTensorFormat\x12!\n\x1dT2A_DFT_TENSOR_FORMAT_UNKNOWN\x10\x00\x12\x1e\n\x1aT2A_WITHOUT_DC_AND_NYQUIST\x10\x01\x12\x14\n\x10T2A_WITH_NYQUIST\x10\x02\x12\x1b\n\x17T2A_WITH_DC_AND_NYQUIST\x10\x03\x32^\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xb0\x93\xf7\xe6\x01 \x01(\x0b\x32*.mediapipe.TensorsToAudioCalculatorOptionsR\x03\x65xtB:Z8github.com/google/mediapipe/mediapipe/calculators/tensor')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.tensor.tensors_to_audio_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_TENSORSTOAUDIOCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z8github.com/google/mediapipe/mediapipe/calculators/tensor'
  _TENSORSTOAUDIOCALCULATOROPTIONS._serialized_start=116
  _TENSORSTOAUDIOCALCULATOROPTIONS._serialized_end=662
  _TENSORSTOAUDIOCALCULATOROPTIONS_DFTTENSORFORMAT._serialized_start=431
  _TENSORSTOAUDIOCALCULATOROPTIONS_DFTTENSORFORMAT._serialized_end=566
# @@protoc_insertion_point(module_scope)
