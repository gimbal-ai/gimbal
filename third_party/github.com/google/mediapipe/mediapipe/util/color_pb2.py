# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/util/color.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1amediapipe/util/color.proto\x12\tmediapipe\"1\n\x05\x43olor\x12\x0c\n\x01r\x18\x01 \x01(\x05R\x01r\x12\x0c\n\x01g\x18\x02 \x01(\x05R\x01g\x12\x0c\n\x01\x62\x18\x03 \x01(\x05R\x01\x62\"\xaa\x01\n\x08\x43olorMap\x12K\n\x0elabel_to_color\x18\x01 \x03(\x0b\x32%.mediapipe.ColorMap.LabelToColorEntryR\x0clabelToColor\x1aQ\n\x11LabelToColorEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12&\n\x05value\x18\x02 \x01(\x0b\x32\x10.mediapipe.ColorR\x05value:\x02\x38\x01\x42Y\n\x1f\x63om.google.mediapipe.util.protoB\nColorProtoZ*github.com/google/mediapipe/mediapipe/util')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.util.color_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\037com.google.mediapipe.util.protoB\nColorProtoZ*github.com/google/mediapipe/mediapipe/util'
  _COLORMAP_LABELTOCOLORENTRY._options = None
  _COLORMAP_LABELTOCOLORENTRY._serialized_options = b'8\001'
  _COLOR._serialized_start=41
  _COLOR._serialized_end=90
  _COLORMAP._serialized_start=93
  _COLORMAP._serialized_end=263
  _COLORMAP_LABELTOCOLORENTRY._serialized_start=182
  _COLORMAP_LABELTOCOLORENTRY._serialized_end=263
# @@protoc_insertion_point(module_scope)
