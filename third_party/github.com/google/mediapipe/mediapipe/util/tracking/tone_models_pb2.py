# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/util/tracking/tone_models.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n)mediapipe/util/tracking/tone_models.proto\x12\tmediapipe\"\xb7\x01\n\rGainBiasModel\x12\x1a\n\x07gain_c1\x18\x01 \x01(\x02:\x01\x31R\x06gainC1\x12\x1a\n\x07\x62ias_c1\x18\x02 \x01(\x02:\x01\x30R\x06\x62iasC1\x12\x1a\n\x07gain_c2\x18\x03 \x01(\x02:\x01\x31R\x06gainC2\x12\x1a\n\x07\x62ias_c2\x18\x04 \x01(\x02:\x01\x30R\x06\x62iasC2\x12\x1a\n\x07gain_c3\x18\x05 \x01(\x02:\x01\x31R\x06gainC3\x12\x1a\n\x07\x62ias_c3\x18\x06 \x01(\x02:\x01\x30R\x06\x62iasC3\"F\n\x14MixtureGainBiasModel\x12.\n\x05model\x18\x01 \x03(\x0b\x32\x18.mediapipe.GainBiasModelR\x05model\"\x99\x02\n\x0f\x41\x66\x66ineToneModel\x12\x14\n\x04g_00\x18\x01 \x01(\x02:\x01\x31R\x03g00\x12\x14\n\x04g_01\x18\x02 \x01(\x02:\x01\x30R\x03g01\x12\x14\n\x04g_02\x18\x03 \x01(\x02:\x01\x30R\x03g02\x12\x14\n\x04g_03\x18\x04 \x01(\x02:\x01\x30R\x03g03\x12\x14\n\x04g_10\x18\x05 \x01(\x02:\x01\x30R\x03g10\x12\x14\n\x04g_11\x18\x06 \x01(\x02:\x01\x31R\x03g11\x12\x14\n\x04g_12\x18\x07 \x01(\x02:\x01\x30R\x03g12\x12\x14\n\x04g_13\x18\x08 \x01(\x02:\x01\x30R\x03g13\x12\x14\n\x04g_20\x18\t \x01(\x02:\x01\x30R\x03g20\x12\x14\n\x04g_21\x18\n \x01(\x02:\x01\x30R\x03g21\x12\x14\n\x04g_22\x18\x0b \x01(\x02:\x01\x31R\x03g22\x12\x14\n\x04g_23\x18\x0c \x01(\x02:\x01\x30R\x03g23\"J\n\x16MixtureAffineToneModel\x12\x30\n\x05model\x18\x01 \x03(\x0b\x32\x1a.mediapipe.AffineToneModelR\x05modelB5Z3github.com/google/mediapipe/mediapipe/util/tracking')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.util.tracking.tone_models_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z3github.com/google/mediapipe/mediapipe/util/tracking'
  _GAINBIASMODEL._serialized_start=57
  _GAINBIASMODEL._serialized_end=240
  _MIXTUREGAINBIASMODEL._serialized_start=242
  _MIXTUREGAINBIASMODEL._serialized_end=312
  _AFFINETONEMODEL._serialized_start=315
  _AFFINETONEMODEL._serialized_end=596
  _MIXTUREAFFINETONEMODEL._serialized_start=598
  _MIXTUREAFFINETONEMODEL._serialized_end=672
# @@protoc_insertion_point(module_scope)
