# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: opentelemetry/proto/common/v1/common.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n*opentelemetry/proto/common/v1/common.proto\x12\x1dopentelemetry.proto.common.v1\"\xe0\x02\n\x08\x41nyValue\x12#\n\x0cstring_value\x18\x01 \x01(\tH\x00R\x0bstringValue\x12\x1f\n\nbool_value\x18\x02 \x01(\x08H\x00R\tboolValue\x12\x1d\n\tint_value\x18\x03 \x01(\x03H\x00R\x08intValue\x12#\n\x0c\x64ouble_value\x18\x04 \x01(\x01H\x00R\x0b\x64oubleValue\x12L\n\x0b\x61rray_value\x18\x05 \x01(\x0b\x32).opentelemetry.proto.common.v1.ArrayValueH\x00R\narrayValue\x12P\n\x0ckvlist_value\x18\x06 \x01(\x0b\x32+.opentelemetry.proto.common.v1.KeyValueListH\x00R\x0bkvlistValue\x12!\n\x0b\x62ytes_value\x18\x07 \x01(\x0cH\x00R\nbytesValueB\x07\n\x05value\"M\n\nArrayValue\x12?\n\x06values\x18\x01 \x03(\x0b\x32\'.opentelemetry.proto.common.v1.AnyValueR\x06values\"O\n\x0cKeyValueList\x12?\n\x06values\x18\x01 \x03(\x0b\x32\'.opentelemetry.proto.common.v1.KeyValueR\x06values\"[\n\x08KeyValue\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12=\n\x05value\x18\x02 \x01(\x0b\x32\'.opentelemetry.proto.common.v1.AnyValueR\x05value\"\xc7\x01\n\x14InstrumentationScope\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x18\n\x07version\x18\x02 \x01(\tR\x07version\x12G\n\nattributes\x18\x03 \x03(\x0b\x32\'.opentelemetry.proto.common.v1.KeyValueR\nattributes\x12\x38\n\x18\x64ropped_attributes_count\x18\x04 \x01(\rR\x16\x64roppedAttributesCountB{\n io.opentelemetry.proto.common.v1B\x0b\x43ommonProtoP\x01Z(go.opentelemetry.io/proto/otlp/common/v1\xaa\x02\x1dOpenTelemetry.Proto.Common.V1b\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'opentelemetry.proto.common.v1.common_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n io.opentelemetry.proto.common.v1B\013CommonProtoP\001Z(go.opentelemetry.io/proto/otlp/common/v1\252\002\035OpenTelemetry.Proto.Common.V1'
  _ANYVALUE._serialized_start=78
  _ANYVALUE._serialized_end=430
  _ARRAYVALUE._serialized_start=432
  _ARRAYVALUE._serialized_end=509
  _KEYVALUELIST._serialized_start=511
  _KEYVALUELIST._serialized_end=590
  _KEYVALUE._serialized_start=592
  _KEYVALUE._serialized_end=683
  _INSTRUMENTATIONSCOPE._serialized_start=686
  _INSTRUMENTATIONSCOPE._serialized_end=885
# @@protoc_insertion_point(module_scope)