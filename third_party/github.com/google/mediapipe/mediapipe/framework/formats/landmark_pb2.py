# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/framework/formats/landmark.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n*mediapipe/framework/formats/landmark.proto\x12\tmediapipe\"p\n\x08Landmark\x12\x0c\n\x01x\x18\x01 \x01(\x02R\x01x\x12\x0c\n\x01y\x18\x02 \x01(\x02R\x01y\x12\x0c\n\x01z\x18\x03 \x01(\x02R\x01z\x12\x1e\n\nvisibility\x18\x04 \x01(\x02R\nvisibility\x12\x1a\n\x08presence\x18\x05 \x01(\x02R\x08presence\"?\n\x0cLandmarkList\x12/\n\x08landmark\x18\x01 \x03(\x0b\x32\x13.mediapipe.LandmarkR\x08landmark\"V\n\x16LandmarkListCollection\x12<\n\rlandmark_list\x18\x01 \x03(\x0b\x32\x17.mediapipe.LandmarkListR\x0clandmarkList\"z\n\x12NormalizedLandmark\x12\x0c\n\x01x\x18\x01 \x01(\x02R\x01x\x12\x0c\n\x01y\x18\x02 \x01(\x02R\x01y\x12\x0c\n\x01z\x18\x03 \x01(\x02R\x01z\x12\x1e\n\nvisibility\x18\x04 \x01(\x02R\nvisibility\x12\x1a\n\x08presence\x18\x05 \x01(\x02R\x08presence\"S\n\x16NormalizedLandmarkList\x12\x39\n\x08landmark\x18\x01 \x03(\x0b\x32\x1d.mediapipe.NormalizedLandmarkR\x08landmark\"j\n NormalizedLandmarkListCollection\x12\x46\n\rlandmark_list\x18\x01 \x03(\x0b\x32!.mediapipe.NormalizedLandmarkListR\x0clandmarkListBl\n\"com.google.mediapipe.formats.protoB\rLandmarkProtoZ7github.com/google/mediapipe/mediapipe/framework/formats')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.framework.formats.landmark_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\"com.google.mediapipe.formats.protoB\rLandmarkProtoZ7github.com/google/mediapipe/mediapipe/framework/formats'
  _LANDMARK._serialized_start=57
  _LANDMARK._serialized_end=169
  _LANDMARKLIST._serialized_start=171
  _LANDMARKLIST._serialized_end=234
  _LANDMARKLISTCOLLECTION._serialized_start=236
  _LANDMARKLISTCOLLECTION._serialized_end=322
  _NORMALIZEDLANDMARK._serialized_start=324
  _NORMALIZEDLANDMARK._serialized_end=446
  _NORMALIZEDLANDMARKLIST._serialized_start=448
  _NORMALIZEDLANDMARKLIST._serialized_end=531
  _NORMALIZEDLANDMARKLISTCOLLECTION._serialized_start=533
  _NORMALIZEDLANDMARKLISTCOLLECTION._serialized_end=639
# @@protoc_insertion_point(module_scope)