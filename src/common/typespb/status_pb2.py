# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/common/typespb/status.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1fsrc/common/typespb/status.proto\x12\tgml.types\x1a\x19google/protobuf/any.proto\"v\n\x06Status\x12*\n\x08\x65rr_code\x18\x01 \x01(\x0e\x32\x0f.gml.types.CodeR\x07\x65rrCode\x12\x10\n\x03msg\x18\x02 \x01(\tR\x03msg\x12.\n\x07\x63ontext\x18\x03 \x01(\x0b\x32\x14.google.protobuf.AnyR\x07\x63ontext*\xe1\x02\n\x04\x43ode\x12\x0b\n\x07\x43ODE_OK\x10\x00\x12\x12\n\x0e\x43ODE_CANCELLED\x10\x01\x12\x10\n\x0c\x43ODE_UNKNOWN\x10\x02\x12\x19\n\x15\x43ODE_INVALID_ARGUMENT\x10\x03\x12\x1a\n\x16\x43ODE_DEADLINE_EXCEEDED\x10\x04\x12\x12\n\x0e\x43ODE_NOT_FOUND\x10\x05\x12\x17\n\x13\x43ODE_ALREADY_EXISTS\x10\x06\x12\x1a\n\x16\x43ODE_PERMISSION_DENIED\x10\x07\x12\x18\n\x14\x43ODE_UNAUTHENTICATED\x10\x08\x12\x11\n\rCODE_INTERNAL\x10\t\x12\x16\n\x12\x43ODE_UNIMPLEMENTED\x10\n\x12\x1d\n\x19\x43ODE_RESOURCE_UNAVAILABLE\x10\x0b\x12\x0f\n\x0b\x43ODE_SYSTEM\x10\x0c\x12\x1c\n\x18\x43ODE_FAILED_PRECONDITION\x10\r\x12\x13\n\x0f\x43ODE_DO_NOT_USE\x10\x64\x42\x31Z/gimletlabs.ai/gimlet/src/common/typespb;typespbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.common.typespb.status_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z/gimletlabs.ai/gimlet/src/common/typespb;typespb'
  _CODE._serialized_start=194
  _CODE._serialized_end=547
  _STATUS._serialized_start=73
  _STATUS._serialized_end=191
# @@protoc_insertion_point(module_scope)