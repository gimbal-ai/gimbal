# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/tensor/tensors_readback_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n>mediapipe/calculators/tensor/tensors_readback_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\x86\x02\n TensorsReadbackCalculatorOptions\x12Z\n\x0ctensor_shape\x18\x01 \x03(\x0b\x32\x37.mediapipe.TensorsReadbackCalculatorOptions.TensorShapeR\x0btensorShape\x1a%\n\x0bTensorShape\x12\x16\n\x04\x64ims\x18\x01 \x03(\x05\x42\x02\x10\x01R\x04\x64ims2_\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xa4\xef\xb9\xf5\x01 \x01(\x0b\x32+.mediapipe.TensorsReadbackCalculatorOptionsR\x03\x65xtB:Z8github.com/google/mediapipe/mediapipe/calculators/tensor')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.tensor.tensors_readback_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_TENSORSREADBACKCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z8github.com/google/mediapipe/mediapipe/calculators/tensor'
  _TENSORSREADBACKCALCULATOROPTIONS_TENSORSHAPE.fields_by_name['dims']._options = None
  _TENSORSREADBACKCALCULATOROPTIONS_TENSORSHAPE.fields_by_name['dims']._serialized_options = b'\020\001'
  _TENSORSREADBACKCALCULATOROPTIONS._serialized_start=116
  _TENSORSREADBACKCALCULATOROPTIONS._serialized_end=378
  _TENSORSREADBACKCALCULATOROPTIONS_TENSORSHAPE._serialized_start=244
  _TENSORSREADBACKCALCULATOROPTIONS_TENSORSHAPE._serialized_end=281
# @@protoc_insertion_point(module_scope)
