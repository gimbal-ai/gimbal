# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/shared/testing/testpb/testing.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\'src/shared/testing/testpb/testing.proto\x12\x0bgml.testing\"\x1f\n\x0bPingRequest\x12\x10\n\x03req\x18\x01 \x01(\tR\x03req\"$\n\x0cPingResponse\x12\x14\n\x05reply\x18\x01 \x01(\tR\x05reply\"+\n\x17PingClientStreamRequest\x12\x10\n\x03req\x18\x01 \x01(\tR\x03req\"0\n\x18PingClientStreamResponse\x12\x14\n\x05reply\x18\x01 \x01(\tR\x05reply\"+\n\x17PingServerStreamRequest\x12\x10\n\x03req\x18\x01 \x01(\tR\x03req\"0\n\x18PingServerStreamResponse\x12\x14\n\x05reply\x18\x01 \x01(\tR\x05reply2\x90\x02\n\x0bPingService\x12;\n\x04Ping\x12\x18.gml.testing.PingRequest\x1a\x19.gml.testing.PingResponse\x12\x61\n\x10PingClientStream\x12$.gml.testing.PingClientStreamRequest\x1a%.gml.testing.PingClientStreamResponse(\x01\x12\x61\n\x10PingServerStream\x12$.gml.testing.PingServerStreamRequest\x1a%.gml.testing.PingServerStreamResponse0\x01\x42\x30Z.gimletlabs.ai/gimlet/src/shared/testing/testpbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.shared.testing.testpb.testing_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z.gimletlabs.ai/gimlet/src/shared/testing/testpb'
  _PINGREQUEST._serialized_start=56
  _PINGREQUEST._serialized_end=87
  _PINGRESPONSE._serialized_start=89
  _PINGRESPONSE._serialized_end=125
  _PINGCLIENTSTREAMREQUEST._serialized_start=127
  _PINGCLIENTSTREAMREQUEST._serialized_end=170
  _PINGCLIENTSTREAMRESPONSE._serialized_start=172
  _PINGCLIENTSTREAMRESPONSE._serialized_end=220
  _PINGSERVERSTREAMREQUEST._serialized_start=222
  _PINGSERVERSTREAMREQUEST._serialized_end=265
  _PINGSERVERSTREAMRESPONSE._serialized_start=267
  _PINGSERVERSTREAMRESPONSE._serialized_end=315
  _PINGSERVICE._serialized_start=318
  _PINGSERVICE._serialized_end=590
# @@protoc_insertion_point(module_scope)