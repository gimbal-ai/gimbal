# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/common/base/testproto/test.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n$src/common/base/testproto/test.proto\x12\ngml.testpb\"1\n\x10TestChildMessage\x12\x1d\n\nstring_val\x18\x01 \x01(\tR\tstringVal\"`\n\x11TestParentMessage\x12\x17\n\x07int_val\x18\x01 \x01(\x03R\x06intVal\x12\x32\n\x05\x63hild\x18\x02 \x03(\x0b\x32\x1c.gml.testpb.TestChildMessageR\x05\x63hildB0Z.gimletlabs.ai/gimlet/src/common/base/testprotob\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.common.base.testproto.test_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z.gimletlabs.ai/gimlet/src/common/base/testproto'
  _TESTCHILDMESSAGE._serialized_start=52
  _TESTCHILDMESSAGE._serialized_end=101
  _TESTPARENTMESSAGE._serialized_start=103
  _TESTPARENTMESSAGE._serialized_end=199
# @@protoc_insertion_point(module_scope)