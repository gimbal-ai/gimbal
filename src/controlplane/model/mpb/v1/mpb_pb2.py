# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/controlplane/model/mpb/v1/mpb.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2
from src.common.typespb import uuid_pb2 as src_dot_common_dot_typespb_dot_uuid__pb2
from src.api.corepb.v1 import model_exec_pb2 as src_dot_api_dot_corepb_dot_v1_dot_model__exec__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\'src/controlplane/model/mpb/v1/mpb.proto\x12\"gml.internal.controlplane.model.v1\x1a\x14gogoproto/gogo.proto\x1a\x1dsrc/common/typespb/uuid.proto\x1a\"src/api/corepb/v1/model_exec.proto\"\x81\x01\n\x0fGetModelRequest\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id\x12\x12\n\x04name\x18\x02 \x01(\tR\x04name\x12\x31\n\x06org_id\x18\x03 \x01(\x0b\x32\x0f.gml.types.UUIDB\t\xe2\xde\x1f\x05OrgIDR\x05orgId\"V\n\x10GetModelResponse\x12\x42\n\nmodel_info\x18\x01 \x01(\x0b\x32#.gml.internal.api.core.v1.ModelInfoR\tmodelInfo\"\x9f\x01\n\x12\x43reateModelRequest\x12\x31\n\x06org_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\t\xe2\xde\x1f\x05OrgIDR\x05orgId\x12\x12\n\x04name\x18\x02 \x01(\tR\x04name\x12\x42\n\nmodel_info\x18\x03 \x01(\x0b\x32#.gml.internal.api.core.v1.ModelInfoR\tmodelInfo\">\n\x13\x43reateModelResponse\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id2\x85\x02\n\x0cModelService\x12u\n\x08GetModel\x12\x33.gml.internal.controlplane.model.v1.GetModelRequest\x1a\x34.gml.internal.controlplane.model.v1.GetModelResponse\x12~\n\x0b\x43reateModel\x12\x36.gml.internal.controlplane.model.v1.CreateModelRequest\x1a\x37.gml.internal.controlplane.model.v1.CreateModelResponseB8Z6gimletlabs.ai/gimlet/src/controlplane/model/mpb/v1;mpbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.controlplane.model.mpb.v1.mpb_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z6gimletlabs.ai/gimlet/src/controlplane/model/mpb/v1;mpb'
  _GETMODELREQUEST.fields_by_name['id']._options = None
  _GETMODELREQUEST.fields_by_name['id']._serialized_options = b'\342\336\037\002ID'
  _GETMODELREQUEST.fields_by_name['org_id']._options = None
  _GETMODELREQUEST.fields_by_name['org_id']._serialized_options = b'\342\336\037\005OrgID'
  _CREATEMODELREQUEST.fields_by_name['org_id']._options = None
  _CREATEMODELREQUEST.fields_by_name['org_id']._serialized_options = b'\342\336\037\005OrgID'
  _CREATEMODELRESPONSE.fields_by_name['id']._options = None
  _CREATEMODELRESPONSE.fields_by_name['id']._serialized_options = b'\342\336\037\002ID'
  _GETMODELREQUEST._serialized_start=169
  _GETMODELREQUEST._serialized_end=298
  _GETMODELRESPONSE._serialized_start=300
  _GETMODELRESPONSE._serialized_end=386
  _CREATEMODELREQUEST._serialized_start=389
  _CREATEMODELREQUEST._serialized_end=548
  _CREATEMODELRESPONSE._serialized_start=550
  _CREATEMODELRESPONSE._serialized_end=612
  _MODELSERVICE._serialized_start=615
  _MODELSERVICE._serialized_end=876
# @@protoc_insertion_point(module_scope)
