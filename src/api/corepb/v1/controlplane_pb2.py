# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/api/corepb/v1/controlplane.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2
from src.common.typespb import uuid_pb2 as src_dot_common_dot_typespb_dot_uuid__pb2
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n$src/api/corepb/v1/controlplane.proto\x12\x18gml.internal.api.core.v1\x1a\x14gogoproto/gogo.proto\x1a\x1dsrc/common/typespb/uuid.proto\x1a\x19google/protobuf/any.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"\xc4\x01\n\nCPMetadata\x12\x37\n\x05topic\x18\x01 \x01(\x0e\x32!.gml.internal.api.core.v1.CPTopicR\x05topic\x12:\n\tentity_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x45ntityIDR\x08\x65ntityId\x12\x41\n\x0erecv_timestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\rrecvTimestamp\"v\n\tCPMessage\x12@\n\x08metadata\x18\x01 \x01(\x0b\x32$.gml.internal.api.core.v1.CPMetadataR\x08metadata\x12\'\n\x03msg\x18\xe8\x07 \x01(\x0b\x32\x14.google.protobuf.AnyR\x03msg\"M\n\x0f\x44\x65viceConnected\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\"J\n\x0c\x44\x65viceUpdate\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\"P\n\x12\x44\x65viceDisconnected\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\"\xd8\x01\n\x1ePhysicalPipelineReconciliation\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\x12Y\n\x14physical_pipeline_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x16\xe2\xde\x1f\x12PhysicalPipelineIDR\x12physicalPipelineId\x12\x1f\n\x0b\x66orce_apply\x18\x03 \x01(\x08R\nforceApply\"\xbc\x01\n PipelineDeploymentReconciliation\x12_\n\x16pipeline_deployment_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x18\xe2\xde\x1f\x14PipelineDeploymentIDR\x14pipelineDeploymentId\x12\x37\n\x08\x66leet_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0b\xe2\xde\x1f\x07\x46leetIDR\x07\x66leetId\"U\n\x17\x42\x61seConfigUpdateRequest\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId*\xf8\x01\n\x07\x43PTopic\x12\x14\n\x10\x43P_TOPIC_UNKNOWN\x10\x00\x12\x1d\n\x19\x43P_TOPIC_DEVICE_CONNECTED\x10\x01\x12-\n)CP_TOPIC_PHYSICAL_PIPELINE_RECONCILIATION\x10\x02\x12 \n\x1c\x43P_TOPIC_DEVICE_DISCONNECTED\x10\x03\x12/\n+CP_TOPIC_PIPELINE_DEPLOYMENT_RECONCILIATION\x10\x04\x12\x1a\n\x16\x43P_TOPIC_DEVICE_UPDATE\x10\x05\x12\x1a\n\x16\x43P_TOPIC_DEVICE_CONFIG\x10\x06\x42/Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.api.corepb.v1.controlplane_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepb'
  _CPMETADATA.fields_by_name['entity_id']._options = None
  _CPMETADATA.fields_by_name['entity_id']._serialized_options = b'\342\336\037\010EntityID'
  _DEVICECONNECTED.fields_by_name['device_id']._options = None
  _DEVICECONNECTED.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _DEVICEUPDATE.fields_by_name['device_id']._options = None
  _DEVICEUPDATE.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _DEVICEDISCONNECTED.fields_by_name['device_id']._options = None
  _DEVICEDISCONNECTED.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _PHYSICALPIPELINERECONCILIATION.fields_by_name['device_id']._options = None
  _PHYSICALPIPELINERECONCILIATION.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _PHYSICALPIPELINERECONCILIATION.fields_by_name['physical_pipeline_id']._options = None
  _PHYSICALPIPELINERECONCILIATION.fields_by_name['physical_pipeline_id']._serialized_options = b'\342\336\037\022PhysicalPipelineID'
  _PIPELINEDEPLOYMENTRECONCILIATION.fields_by_name['pipeline_deployment_id']._options = None
  _PIPELINEDEPLOYMENTRECONCILIATION.fields_by_name['pipeline_deployment_id']._serialized_options = b'\342\336\037\024PipelineDeploymentID'
  _PIPELINEDEPLOYMENTRECONCILIATION.fields_by_name['fleet_id']._options = None
  _PIPELINEDEPLOYMENTRECONCILIATION.fields_by_name['fleet_id']._serialized_options = b'\342\336\037\007FleetID'
  _BASECONFIGUPDATEREQUEST.fields_by_name['device_id']._options = None
  _BASECONFIGUPDATEREQUEST.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _CPTOPIC._serialized_start=1233
  _CPTOPIC._serialized_end=1481
  _CPMETADATA._serialized_start=180
  _CPMETADATA._serialized_end=376
  _CPMESSAGE._serialized_start=378
  _CPMESSAGE._serialized_end=496
  _DEVICECONNECTED._serialized_start=498
  _DEVICECONNECTED._serialized_end=575
  _DEVICEUPDATE._serialized_start=577
  _DEVICEUPDATE._serialized_end=651
  _DEVICEDISCONNECTED._serialized_start=653
  _DEVICEDISCONNECTED._serialized_end=733
  _PHYSICALPIPELINERECONCILIATION._serialized_start=736
  _PHYSICALPIPELINERECONCILIATION._serialized_end=952
  _PIPELINEDEPLOYMENTRECONCILIATION._serialized_start=955
  _PIPELINEDEPLOYMENTRECONCILIATION._serialized_end=1143
  _BASECONFIGUPDATEREQUEST._serialized_start=1145
  _BASECONFIGUPDATEREQUEST._serialized_end=1230
# @@protoc_insertion_point(module_scope)
