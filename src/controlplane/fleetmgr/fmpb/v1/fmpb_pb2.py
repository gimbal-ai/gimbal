# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/controlplane/fleetmgr/fmpb/v1/fmpb.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from src.api.corepb.v1 import cp_edge_pb2 as src_dot_api_dot_corepb_dot_v1_dot_cp__edge__pb2
from src.common.typespb import uuid_pb2 as src_dot_common_dot_typespb_dot_uuid__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n,src/controlplane/fleetmgr/fmpb/v1/fmpb.proto\x12%gml.internal.controlplane.fleetmgr.v1\x1a\x14gogoproto/gogo.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1fsrc/api/corepb/v1/cp_edge.proto\x1a\x1dsrc/common/typespb/uuid.proto\"\xb7\x03\n\tFleetInfo\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id\x12@\n\x06org_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x18\xe2\xde\x1f\x05OrgID\xf2\xde\x1f\x0b\x64\x62:\"org_id\"R\x05orgId\x12\x12\n\x04name\x18\x03 \x01(\tR\x04name\x12 \n\x0b\x64\x65scription\x18\x05 \x01(\tR\x0b\x64\x65scription\x12N\n\ncreated_at\x18\x06 \x01(\x0b\x32\x1a.google.protobuf.TimestampB\x13\xf2\xde\x1f\x0f\x64\x62:\"created_at\"R\tcreatedAt\x12N\n\x04tags\x18\x07 \x03(\x0b\x32:.gml.internal.controlplane.fleetmgr.v1.FleetInfo.TagsEntryR\x04tags\x1a\x63\n\tTagsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12@\n\x05value\x18\x02 \x01(\x0b\x32*.gml.internal.controlplane.fleetmgr.v1.TagR\x05value:\x02\x38\x01J\x04\x08\x04\x10\x05\"\\\n\x12\x43reateFleetRequest\x12\x46\n\x05\x66leet\x18\x01 \x01(\x0b\x32\x30.gml.internal.controlplane.fleetmgr.v1.FleetInfoR\x05\x66leet\"]\n\x13\x43reateFleetResponse\x12\x46\n\x05\x66leet\x18\x01 \x01(\x0b\x32\x30.gml.internal.controlplane.fleetmgr.v1.FleetInfoR\x05\x66leet\"@\n\x0fGetFleetRequest\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02idJ\x04\x08\x02\x10\x03\"Z\n\x10GetFleetResponse\x12\x46\n\x05\x66leet\x18\x01 \x01(\x0b\x32\x30.gml.internal.controlplane.fleetmgr.v1.FleetInfoR\x05\x66leet\"^\n\x15GetFleetByNameRequest\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x31\n\x06org_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\t\xe2\xde\x1f\x05OrgIDR\x05orgId\"`\n\x16GetFleetByNameResponse\x12\x46\n\x05\x66leet\x18\x01 \x01(\x0b\x32\x30.gml.internal.controlplane.fleetmgr.v1.FleetInfoR\x05\x66leet\"F\n\x11ListFleetsRequest\x12\x31\n\x06org_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\t\xe2\xde\x1f\x05OrgIDR\x05orgId\"^\n\x12ListFleetsResponse\x12H\n\x06\x66leets\x18\x01 \x03(\x0b\x32\x30.gml.internal.controlplane.fleetmgr.v1.FleetInfoR\x06\x66leets\"\x86\x01\n\x12UpdateFleetRequest\x12\x46\n\x05\x66leet\x18\x01 \x01(\x0b\x32\x30.gml.internal.controlplane.fleetmgr.v1.FleetInfoR\x05\x66leet\x12(\n\x10\x64\x65leted_tag_keys\x18\x02 \x03(\tR\x0e\x64\x65letedTagKeys\"]\n\x13UpdateFleetResponse\x12\x46\n\x05\x66leet\x18\x01 \x01(\x0b\x32\x30.gml.internal.controlplane.fleetmgr.v1.FleetInfoR\x05\x66leet\"}\n\x03Tag\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value\x12N\n\x08metadata\x18\x03 \x01(\x0b\x32\x32.gml.internal.controlplane.fleetmgr.v1.TagMetadataR\x08metadata\"\xe1\x01\n\x0bTagMetadata\x12\x32\n\x0cis_inherited\x18\x01 \x01(\x08\x42\x0f\xe2\xde\x1f\x0bIsInheritedR\x0bisInherited\x12N\n\nupdated_at\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.TimestampB\x13\xf2\xde\x1f\x0f\x64\x62:\"updated_at\"R\tupdatedAt\x12N\n\ncreated_at\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.TimestampB\x13\xf2\xde\x1f\x0f\x64\x62:\"created_at\"R\tcreatedAt\"P\n\x15GetDefaultTagsRequest\x12\x37\n\x08\x66leet_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0b\xe2\xde\x1f\x07\x46leetIDR\x07\x66leetId\"\xda\x01\n\x16GetDefaultTagsResponse\x12[\n\x04tags\x18\x01 \x03(\x0b\x32G.gml.internal.controlplane.fleetmgr.v1.GetDefaultTagsResponse.TagsEntryR\x04tags\x1a\x63\n\tTagsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12@\n\x05value\x18\x02 \x01(\x0b\x32*.gml.internal.controlplane.fleetmgr.v1.TagR\x05value:\x02\x38\x01\"\x90\x01\n\x17UpsertDefaultTagRequest\x12\x37\n\x08\x66leet_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0b\xe2\xde\x1f\x07\x46leetIDR\x07\x66leetId\x12<\n\x03tag\x18\x02 \x01(\x0b\x32*.gml.internal.controlplane.fleetmgr.v1.TagR\x03tag\"\x1a\n\x18UpsertDefaultTagResponse\"d\n\x17\x44\x65leteDefaultTagRequest\x12\x37\n\x08\x66leet_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0b\xe2\xde\x1f\x07\x46leetIDR\x07\x66leetId\x12\x10\n\x03key\x18\x02 \x01(\tR\x03key\"\x1a\n\x18\x44\x65leteDefaultTagResponse\"e\n\x06OSInfo\x12\x41\n\x04kind\x18\x01 \x01(\x0e\x32-.gml.internal.controlplane.fleetmgr.v1.OSKindR\x04kind\x12\x18\n\x07version\x18\x02 \x01(\tR\x07version\"\xd3\x05\n\nDeviceInfo\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id\x12\x16\n\x06serial\x18\x02 \x01(\tR\x06serial\x12\x1a\n\x08hostname\x18\x03 \x01(\tR\x08hostname\x12!\n\x0c\x64isplay_name\x18\x0b \x01(\tR\x0b\x64isplayName\x12H\n\x08\x66leet_id\x18\x04 \x01(\x0b\x32\x0f.gml.types.UUIDB\x1c\xe2\xde\x1f\x07\x46leetID\xf2\xde\x1f\rdb:\"fleet_id\"R\x07\x66leetId\x12Y\n\x11last_heartbeat_ns\x18\x05 \x01(\x03\x42-\xe2\xde\x1f\x0fLastHeartbeatNS\xf2\xde\x1f\x16\x64\x62:\"last_heartbeat_ns\"R\x0flastHeartbeatNs\x12K\n\x06status\x18\x06 \x01(\x0e\x32\x33.gml.internal.controlplane.fleetmgr.v1.DeviceStatusR\x06status\x12P\n\x0c\x63\x61pabilities\x18\x08 \x01(\x0b\x32,.gml.internal.api.core.v1.DeviceCapabilitiesR\x0c\x63\x61pabilities\x12O\n\x04tags\x18\t \x03(\x0b\x32;.gml.internal.controlplane.fleetmgr.v1.DeviceInfo.TagsEntryR\x04tags\x12\x45\n\x02os\x18\n \x01(\x0b\x32-.gml.internal.controlplane.fleetmgr.v1.OSInfoB\x06\xe2\xde\x1f\x02OSR\x02os\x1a\x63\n\tTagsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12@\n\x05value\x18\x02 \x01(\x0b\x32*.gml.internal.controlplane.fleetmgr.v1.TagR\x05value:\x02\x38\x01J\x04\x08\x07\x10\x08\"k\n#UnassociateTagsWithDeployKeyRequest\x12\x44\n\rdeploy_key_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0f\xe2\xde\x1f\x0b\x44\x65ployKeyIDR\x0b\x64\x65ployKeyId\"&\n$UnassociateTagsWithDeployKeyResponse\"\xc3\x02\n!AssociateTagsWithDeployKeyRequest\x12\x44\n\rdeploy_key_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0f\xe2\xde\x1f\x0b\x44\x65ployKeyIDR\x0b\x64\x65ployKeyId\x12\x66\n\x04tags\x18\x02 \x03(\x0b\x32R.gml.internal.controlplane.fleetmgr.v1.AssociateTagsWithDeployKeyRequest.TagsEntryR\x04tags\x12\x37\n\x08\x66leet_id\x18\x03 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0b\xe2\xde\x1f\x07\x46leetIDR\x07\x66leetId\x1a\x37\n\tTagsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\"\xb8\x02\n\"AssociateTagsWithDeployKeyResponse\x12\x44\n\rdeploy_key_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0f\xe2\xde\x1f\x0b\x44\x65ployKeyIDR\x0b\x64\x65ployKeyId\x12g\n\x04tags\x18\x02 \x03(\x0b\x32S.gml.internal.controlplane.fleetmgr.v1.AssociateTagsWithDeployKeyResponse.TagsEntryR\x04tags\x1a\x63\n\tTagsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12@\n\x05value\x18\x02 \x01(\x0b\x32*.gml.internal.controlplane.fleetmgr.v1.TagR\x05value:\x02\x38\x01\"n\n&ListTagsAssociatedWithDeployKeyRequest\x12\x44\n\rdeploy_key_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0f\xe2\xde\x1f\x0b\x44\x65ployKeyIDR\x0b\x64\x65ployKeyId\"\xc2\x02\n\'ListTagsAssociatedWithDeployKeyResponse\x12\x44\n\rdeploy_key_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0f\xe2\xde\x1f\x0b\x44\x65ployKeyIDR\x0b\x64\x65ployKeyId\x12l\n\x04tags\x18\x02 \x03(\x0b\x32X.gml.internal.controlplane.fleetmgr.v1.ListTagsAssociatedWithDeployKeyResponse.TagsEntryR\x04tags\x1a\x63\n\tTagsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12@\n\x05value\x18\x02 \x01(\x0b\x32*.gml.internal.controlplane.fleetmgr.v1.TagR\x05value:\x02\x38\x01\"\x99\x01\n\x0fRegisterRequest\x12#\n\rdevice_serial\x18\x01 \x01(\tR\x0c\x64\x65viceSerial\x12\x1a\n\x08hostname\x18\x02 \x01(\tR\x08hostname\x12\x45\n\x02os\x18\x03 \x01(\x0b\x32-.gml.internal.controlplane.fleetmgr.v1.OSInfoB\x06\xe2\xde\x1f\x02OSR\x02os\"N\n\x10RegisterResponse\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\"Q\n\x13UpdateStatusRequest\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\"\x16\n\x14UpdateStatusResponse\";\n\x10GetDeviceRequest\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id\"^\n\x11GetDeviceResponse\x12I\n\x06\x64\x65vice\x18\x01 \x01(\x0b\x32\x31.gml.internal.controlplane.fleetmgr.v1.DeviceInfoR\x06\x64\x65vice\"M\n\x12ListDevicesRequest\x12\x37\n\x08\x66leet_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0b\xe2\xde\x1f\x07\x46leetIDR\x07\x66leetId\"b\n\x13ListDevicesResponse\x12K\n\x07\x64\x65vices\x18\x01 \x03(\x0b\x32\x31.gml.internal.controlplane.fleetmgr.v1.DeviceInfoR\x07\x64\x65vices\"\x8a\x01\n\x13UpdateDeviceRequest\x12I\n\x06\x64\x65vice\x18\x01 \x01(\x0b\x32\x31.gml.internal.controlplane.fleetmgr.v1.DeviceInfoR\x06\x64\x65vice\x12(\n\x10\x64\x65leted_tag_keys\x18\x02 \x03(\tR\x0e\x64\x65letedTagKeys\"a\n\x14UpdateDeviceResponse\x12I\n\x06\x64\x65vice\x18\x01 \x01(\x0b\x32\x31.gml.internal.controlplane.fleetmgr.v1.DeviceInfoR\x06\x64\x65vice\"B\n\x14\x44\x65leteDevicesRequest\x12*\n\x03ids\x18\x01 \x03(\x0b\x32\x0f.gml.types.UUIDB\x07\xe2\xde\x1f\x03IDsR\x03ids\"\x17\n\x15\x44\x65leteDevicesResponse\"\xac\x01\n\x1cSetDeviceCapabilitiesRequest\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\x12P\n\x0c\x63\x61pabilities\x18\x02 \x01(\x0b\x32,.gml.internal.api.core.v1.DeviceCapabilitiesR\x0c\x63\x61pabilities\"\x1f\n\x1dSetDeviceCapabilitiesResponse\"L\n\x0eGetTagsRequest\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\"\xcc\x01\n\x0fGetTagsResponse\x12T\n\x04tags\x18\x01 \x03(\x0b\x32@.gml.internal.controlplane.fleetmgr.v1.GetTagsResponse.TagsEntryR\x04tags\x1a\x63\n\tTagsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12@\n\x05value\x18\x02 \x01(\x0b\x32*.gml.internal.controlplane.fleetmgr.v1.TagR\x05value:\x02\x38\x01\"\x8c\x01\n\x10UpsertTagRequest\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\x12<\n\x03tag\x18\x02 \x01(\x0b\x32*.gml.internal.controlplane.fleetmgr.v1.TagR\x03tag\"\x13\n\x11UpsertTagResponse\"`\n\x10\x44\x65leteTagRequest\x12:\n\tdevice_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\x12\x10\n\x03key\x18\x02 \x01(\tR\x03key\"\x13\n\x11\x44\x65leteTagResponse*\x81\x01\n\x0c\x44\x65viceStatus\x12\x19\n\x15\x44\x45VICE_STATUS_UNKNOWN\x10\x00\x12\x19\n\x15\x44\x45VICE_STATUS_HEALTHY\x10\x01\x12\x1b\n\x17\x44\x45VICE_STATUS_UNHEALTHY\x10\x02\x12\x1e\n\x1a\x44\x45VICE_STATUS_DISCONNECTED\x10\x03*\x91\x01\n\x06OSKind\x12\x13\n\x0fOS_KIND_UNKNOWN\x10\x00\x12\x11\n\rOS_KIND_LINUX\x10\x01\x12\x13\n\x0fOS_KIND_WINDOWS\x10\x02\x12\x11\n\rOS_KIND_MACOS\x10\x03\x12\x0f\n\x0bOS_KIND_IOS\x10\x04\x12\x13\n\x0fOS_KIND_ANDROID\x10\x05\x12\x11\n\rOS_KIND_OTHER\x10\x06\x32\xec\x08\n\x0f\x46leetMgrService\x12\x84\x01\n\x0b\x43reateFleet\x12\x39.gml.internal.controlplane.fleetmgr.v1.CreateFleetRequest\x1a:.gml.internal.controlplane.fleetmgr.v1.CreateFleetResponse\x12{\n\x08GetFleet\x12\x36.gml.internal.controlplane.fleetmgr.v1.GetFleetRequest\x1a\x37.gml.internal.controlplane.fleetmgr.v1.GetFleetResponse\x12\x8d\x01\n\x0eGetFleetByName\x12<.gml.internal.controlplane.fleetmgr.v1.GetFleetByNameRequest\x1a=.gml.internal.controlplane.fleetmgr.v1.GetFleetByNameResponse\x12\x81\x01\n\nListFleets\x12\x38.gml.internal.controlplane.fleetmgr.v1.ListFleetsRequest\x1a\x39.gml.internal.controlplane.fleetmgr.v1.ListFleetsResponse\x12\x84\x01\n\x0bUpdateFleet\x12\x39.gml.internal.controlplane.fleetmgr.v1.UpdateFleetRequest\x1a:.gml.internal.controlplane.fleetmgr.v1.UpdateFleetResponse\x12\x8d\x01\n\x0eGetDefaultTags\x12<.gml.internal.controlplane.fleetmgr.v1.GetDefaultTagsRequest\x1a=.gml.internal.controlplane.fleetmgr.v1.GetDefaultTagsResponse\x12\x93\x01\n\x10UpsertDefaultTag\x12>.gml.internal.controlplane.fleetmgr.v1.UpsertDefaultTagRequest\x1a?.gml.internal.controlplane.fleetmgr.v1.UpsertDefaultTagResponse\x12\x93\x01\n\x10\x44\x65leteDefaultTag\x12>.gml.internal.controlplane.fleetmgr.v1.DeleteDefaultTagRequest\x1a?.gml.internal.controlplane.fleetmgr.v1.DeleteDefaultTagResponse2\x92\x01\n\x13\x46leetMgrEdgeService\x12{\n\x08Register\x12\x36.gml.internal.controlplane.fleetmgr.v1.RegisterRequest\x1a\x37.gml.internal.controlplane.fleetmgr.v1.RegisterResponse2\xe4\x06\n\x15\x46leetMgrDeviceService\x12\x87\x01\n\x0cUpdateStatus\x12:.gml.internal.controlplane.fleetmgr.v1.UpdateStatusRequest\x1a;.gml.internal.controlplane.fleetmgr.v1.UpdateStatusResponse\x12~\n\tGetDevice\x12\x37.gml.internal.controlplane.fleetmgr.v1.GetDeviceRequest\x1a\x38.gml.internal.controlplane.fleetmgr.v1.GetDeviceResponse\x12\x84\x01\n\x0bListDevices\x12\x39.gml.internal.controlplane.fleetmgr.v1.ListDevicesRequest\x1a:.gml.internal.controlplane.fleetmgr.v1.ListDevicesResponse\x12\x87\x01\n\x0cUpdateDevice\x12:.gml.internal.controlplane.fleetmgr.v1.UpdateDeviceRequest\x1a;.gml.internal.controlplane.fleetmgr.v1.UpdateDeviceResponse\x12\x8a\x01\n\rDeleteDevices\x12;.gml.internal.controlplane.fleetmgr.v1.DeleteDevicesRequest\x1a<.gml.internal.controlplane.fleetmgr.v1.DeleteDevicesResponse\x12\xa2\x01\n\x15SetDeviceCapabilities\x12\x43.gml.internal.controlplane.fleetmgr.v1.SetDeviceCapabilitiesRequest\x1a\x44.gml.internal.controlplane.fleetmgr.v1.SetDeviceCapabilitiesResponse2\xc6\x07\n\x19\x46leetMgrDeviceTagsService\x12x\n\x07GetTags\x12\x35.gml.internal.controlplane.fleetmgr.v1.GetTagsRequest\x1a\x36.gml.internal.controlplane.fleetmgr.v1.GetTagsResponse\x12~\n\tUpsertTag\x12\x37.gml.internal.controlplane.fleetmgr.v1.UpsertTagRequest\x1a\x38.gml.internal.controlplane.fleetmgr.v1.UpsertTagResponse\x12~\n\tDeleteTag\x12\x37.gml.internal.controlplane.fleetmgr.v1.DeleteTagRequest\x1a\x38.gml.internal.controlplane.fleetmgr.v1.DeleteTagResponse\x12\xb1\x01\n\x1a\x41ssociateTagsWithDeployKey\x12H.gml.internal.controlplane.fleetmgr.v1.AssociateTagsWithDeployKeyRequest\x1aI.gml.internal.controlplane.fleetmgr.v1.AssociateTagsWithDeployKeyResponse\x12\xb7\x01\n\x1cUnassociateTagsWithDeployKey\x12J.gml.internal.controlplane.fleetmgr.v1.UnassociateTagsWithDeployKeyRequest\x1aK.gml.internal.controlplane.fleetmgr.v1.UnassociateTagsWithDeployKeyResponse\x12\xc0\x01\n\x1fListTagsAssociatedWithDeployKey\x12M.gml.internal.controlplane.fleetmgr.v1.ListTagsAssociatedWithDeployKeyRequest\x1aN.gml.internal.controlplane.fleetmgr.v1.ListTagsAssociatedWithDeployKeyResponseB=Z;gimletlabs.ai/gimlet/src/controlplane/fleetmgr/fmpb/v1;fmpbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.controlplane.fleetmgr.fmpb.v1.fmpb_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z;gimletlabs.ai/gimlet/src/controlplane/fleetmgr/fmpb/v1;fmpb'
  _FLEETINFO_TAGSENTRY._options = None
  _FLEETINFO_TAGSENTRY._serialized_options = b'8\001'
  _FLEETINFO.fields_by_name['id']._options = None
  _FLEETINFO.fields_by_name['id']._serialized_options = b'\342\336\037\002ID'
  _FLEETINFO.fields_by_name['org_id']._options = None
  _FLEETINFO.fields_by_name['org_id']._serialized_options = b'\342\336\037\005OrgID\362\336\037\013db:\"org_id\"'
  _FLEETINFO.fields_by_name['created_at']._options = None
  _FLEETINFO.fields_by_name['created_at']._serialized_options = b'\362\336\037\017db:\"created_at\"'
  _GETFLEETREQUEST.fields_by_name['id']._options = None
  _GETFLEETREQUEST.fields_by_name['id']._serialized_options = b'\342\336\037\002ID'
  _GETFLEETBYNAMEREQUEST.fields_by_name['org_id']._options = None
  _GETFLEETBYNAMEREQUEST.fields_by_name['org_id']._serialized_options = b'\342\336\037\005OrgID'
  _LISTFLEETSREQUEST.fields_by_name['org_id']._options = None
  _LISTFLEETSREQUEST.fields_by_name['org_id']._serialized_options = b'\342\336\037\005OrgID'
  _TAGMETADATA.fields_by_name['is_inherited']._options = None
  _TAGMETADATA.fields_by_name['is_inherited']._serialized_options = b'\342\336\037\013IsInherited'
  _TAGMETADATA.fields_by_name['updated_at']._options = None
  _TAGMETADATA.fields_by_name['updated_at']._serialized_options = b'\362\336\037\017db:\"updated_at\"'
  _TAGMETADATA.fields_by_name['created_at']._options = None
  _TAGMETADATA.fields_by_name['created_at']._serialized_options = b'\362\336\037\017db:\"created_at\"'
  _GETDEFAULTTAGSREQUEST.fields_by_name['fleet_id']._options = None
  _GETDEFAULTTAGSREQUEST.fields_by_name['fleet_id']._serialized_options = b'\342\336\037\007FleetID'
  _GETDEFAULTTAGSRESPONSE_TAGSENTRY._options = None
  _GETDEFAULTTAGSRESPONSE_TAGSENTRY._serialized_options = b'8\001'
  _UPSERTDEFAULTTAGREQUEST.fields_by_name['fleet_id']._options = None
  _UPSERTDEFAULTTAGREQUEST.fields_by_name['fleet_id']._serialized_options = b'\342\336\037\007FleetID'
  _DELETEDEFAULTTAGREQUEST.fields_by_name['fleet_id']._options = None
  _DELETEDEFAULTTAGREQUEST.fields_by_name['fleet_id']._serialized_options = b'\342\336\037\007FleetID'
  _DEVICEINFO_TAGSENTRY._options = None
  _DEVICEINFO_TAGSENTRY._serialized_options = b'8\001'
  _DEVICEINFO.fields_by_name['id']._options = None
  _DEVICEINFO.fields_by_name['id']._serialized_options = b'\342\336\037\002ID'
  _DEVICEINFO.fields_by_name['fleet_id']._options = None
  _DEVICEINFO.fields_by_name['fleet_id']._serialized_options = b'\342\336\037\007FleetID\362\336\037\rdb:\"fleet_id\"'
  _DEVICEINFO.fields_by_name['last_heartbeat_ns']._options = None
  _DEVICEINFO.fields_by_name['last_heartbeat_ns']._serialized_options = b'\342\336\037\017LastHeartbeatNS\362\336\037\026db:\"last_heartbeat_ns\"'
  _DEVICEINFO.fields_by_name['os']._options = None
  _DEVICEINFO.fields_by_name['os']._serialized_options = b'\342\336\037\002OS'
  _UNASSOCIATETAGSWITHDEPLOYKEYREQUEST.fields_by_name['deploy_key_id']._options = None
  _UNASSOCIATETAGSWITHDEPLOYKEYREQUEST.fields_by_name['deploy_key_id']._serialized_options = b'\342\336\037\013DeployKeyID'
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST_TAGSENTRY._options = None
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST_TAGSENTRY._serialized_options = b'8\001'
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST.fields_by_name['deploy_key_id']._options = None
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST.fields_by_name['deploy_key_id']._serialized_options = b'\342\336\037\013DeployKeyID'
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST.fields_by_name['fleet_id']._options = None
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST.fields_by_name['fleet_id']._serialized_options = b'\342\336\037\007FleetID'
  _ASSOCIATETAGSWITHDEPLOYKEYRESPONSE_TAGSENTRY._options = None
  _ASSOCIATETAGSWITHDEPLOYKEYRESPONSE_TAGSENTRY._serialized_options = b'8\001'
  _ASSOCIATETAGSWITHDEPLOYKEYRESPONSE.fields_by_name['deploy_key_id']._options = None
  _ASSOCIATETAGSWITHDEPLOYKEYRESPONSE.fields_by_name['deploy_key_id']._serialized_options = b'\342\336\037\013DeployKeyID'
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYREQUEST.fields_by_name['deploy_key_id']._options = None
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYREQUEST.fields_by_name['deploy_key_id']._serialized_options = b'\342\336\037\013DeployKeyID'
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYRESPONSE_TAGSENTRY._options = None
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYRESPONSE_TAGSENTRY._serialized_options = b'8\001'
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYRESPONSE.fields_by_name['deploy_key_id']._options = None
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYRESPONSE.fields_by_name['deploy_key_id']._serialized_options = b'\342\336\037\013DeployKeyID'
  _REGISTERREQUEST.fields_by_name['os']._options = None
  _REGISTERREQUEST.fields_by_name['os']._serialized_options = b'\342\336\037\002OS'
  _REGISTERRESPONSE.fields_by_name['device_id']._options = None
  _REGISTERRESPONSE.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _UPDATESTATUSREQUEST.fields_by_name['device_id']._options = None
  _UPDATESTATUSREQUEST.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _GETDEVICEREQUEST.fields_by_name['id']._options = None
  _GETDEVICEREQUEST.fields_by_name['id']._serialized_options = b'\342\336\037\002ID'
  _LISTDEVICESREQUEST.fields_by_name['fleet_id']._options = None
  _LISTDEVICESREQUEST.fields_by_name['fleet_id']._serialized_options = b'\342\336\037\007FleetID'
  _DELETEDEVICESREQUEST.fields_by_name['ids']._options = None
  _DELETEDEVICESREQUEST.fields_by_name['ids']._serialized_options = b'\342\336\037\003IDs'
  _SETDEVICECAPABILITIESREQUEST.fields_by_name['device_id']._options = None
  _SETDEVICECAPABILITIESREQUEST.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _GETTAGSREQUEST.fields_by_name['device_id']._options = None
  _GETTAGSREQUEST.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _GETTAGSRESPONSE_TAGSENTRY._options = None
  _GETTAGSRESPONSE_TAGSENTRY._serialized_options = b'8\001'
  _UPSERTTAGREQUEST.fields_by_name['device_id']._options = None
  _UPSERTTAGREQUEST.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _DELETETAGREQUEST.fields_by_name['device_id']._options = None
  _DELETETAGREQUEST.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _DEVICESTATUS._serialized_start=6397
  _DEVICESTATUS._serialized_end=6526
  _OSKIND._serialized_start=6529
  _OSKIND._serialized_end=6674
  _FLEETINFO._serialized_start=207
  _FLEETINFO._serialized_end=646
  _FLEETINFO_TAGSENTRY._serialized_start=541
  _FLEETINFO_TAGSENTRY._serialized_end=640
  _CREATEFLEETREQUEST._serialized_start=648
  _CREATEFLEETREQUEST._serialized_end=740
  _CREATEFLEETRESPONSE._serialized_start=742
  _CREATEFLEETRESPONSE._serialized_end=835
  _GETFLEETREQUEST._serialized_start=837
  _GETFLEETREQUEST._serialized_end=901
  _GETFLEETRESPONSE._serialized_start=903
  _GETFLEETRESPONSE._serialized_end=993
  _GETFLEETBYNAMEREQUEST._serialized_start=995
  _GETFLEETBYNAMEREQUEST._serialized_end=1089
  _GETFLEETBYNAMERESPONSE._serialized_start=1091
  _GETFLEETBYNAMERESPONSE._serialized_end=1187
  _LISTFLEETSREQUEST._serialized_start=1189
  _LISTFLEETSREQUEST._serialized_end=1259
  _LISTFLEETSRESPONSE._serialized_start=1261
  _LISTFLEETSRESPONSE._serialized_end=1355
  _UPDATEFLEETREQUEST._serialized_start=1358
  _UPDATEFLEETREQUEST._serialized_end=1492
  _UPDATEFLEETRESPONSE._serialized_start=1494
  _UPDATEFLEETRESPONSE._serialized_end=1587
  _TAG._serialized_start=1589
  _TAG._serialized_end=1714
  _TAGMETADATA._serialized_start=1717
  _TAGMETADATA._serialized_end=1942
  _GETDEFAULTTAGSREQUEST._serialized_start=1944
  _GETDEFAULTTAGSREQUEST._serialized_end=2024
  _GETDEFAULTTAGSRESPONSE._serialized_start=2027
  _GETDEFAULTTAGSRESPONSE._serialized_end=2245
  _GETDEFAULTTAGSRESPONSE_TAGSENTRY._serialized_start=541
  _GETDEFAULTTAGSRESPONSE_TAGSENTRY._serialized_end=640
  _UPSERTDEFAULTTAGREQUEST._serialized_start=2248
  _UPSERTDEFAULTTAGREQUEST._serialized_end=2392
  _UPSERTDEFAULTTAGRESPONSE._serialized_start=2394
  _UPSERTDEFAULTTAGRESPONSE._serialized_end=2420
  _DELETEDEFAULTTAGREQUEST._serialized_start=2422
  _DELETEDEFAULTTAGREQUEST._serialized_end=2522
  _DELETEDEFAULTTAGRESPONSE._serialized_start=2524
  _DELETEDEFAULTTAGRESPONSE._serialized_end=2550
  _OSINFO._serialized_start=2552
  _OSINFO._serialized_end=2653
  _DEVICEINFO._serialized_start=2656
  _DEVICEINFO._serialized_end=3379
  _DEVICEINFO_TAGSENTRY._serialized_start=541
  _DEVICEINFO_TAGSENTRY._serialized_end=640
  _UNASSOCIATETAGSWITHDEPLOYKEYREQUEST._serialized_start=3381
  _UNASSOCIATETAGSWITHDEPLOYKEYREQUEST._serialized_end=3488
  _UNASSOCIATETAGSWITHDEPLOYKEYRESPONSE._serialized_start=3490
  _UNASSOCIATETAGSWITHDEPLOYKEYRESPONSE._serialized_end=3528
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST._serialized_start=3531
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST._serialized_end=3854
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST_TAGSENTRY._serialized_start=3799
  _ASSOCIATETAGSWITHDEPLOYKEYREQUEST_TAGSENTRY._serialized_end=3854
  _ASSOCIATETAGSWITHDEPLOYKEYRESPONSE._serialized_start=3857
  _ASSOCIATETAGSWITHDEPLOYKEYRESPONSE._serialized_end=4169
  _ASSOCIATETAGSWITHDEPLOYKEYRESPONSE_TAGSENTRY._serialized_start=541
  _ASSOCIATETAGSWITHDEPLOYKEYRESPONSE_TAGSENTRY._serialized_end=640
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYREQUEST._serialized_start=4171
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYREQUEST._serialized_end=4281
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYRESPONSE._serialized_start=4284
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYRESPONSE._serialized_end=4606
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYRESPONSE_TAGSENTRY._serialized_start=541
  _LISTTAGSASSOCIATEDWITHDEPLOYKEYRESPONSE_TAGSENTRY._serialized_end=640
  _REGISTERREQUEST._serialized_start=4609
  _REGISTERREQUEST._serialized_end=4762
  _REGISTERRESPONSE._serialized_start=4764
  _REGISTERRESPONSE._serialized_end=4842
  _UPDATESTATUSREQUEST._serialized_start=4844
  _UPDATESTATUSREQUEST._serialized_end=4925
  _UPDATESTATUSRESPONSE._serialized_start=4927
  _UPDATESTATUSRESPONSE._serialized_end=4949
  _GETDEVICEREQUEST._serialized_start=4951
  _GETDEVICEREQUEST._serialized_end=5010
  _GETDEVICERESPONSE._serialized_start=5012
  _GETDEVICERESPONSE._serialized_end=5106
  _LISTDEVICESREQUEST._serialized_start=5108
  _LISTDEVICESREQUEST._serialized_end=5185
  _LISTDEVICESRESPONSE._serialized_start=5187
  _LISTDEVICESRESPONSE._serialized_end=5285
  _UPDATEDEVICEREQUEST._serialized_start=5288
  _UPDATEDEVICEREQUEST._serialized_end=5426
  _UPDATEDEVICERESPONSE._serialized_start=5428
  _UPDATEDEVICERESPONSE._serialized_end=5525
  _DELETEDEVICESREQUEST._serialized_start=5527
  _DELETEDEVICESREQUEST._serialized_end=5593
  _DELETEDEVICESRESPONSE._serialized_start=5595
  _DELETEDEVICESRESPONSE._serialized_end=5618
  _SETDEVICECAPABILITIESREQUEST._serialized_start=5621
  _SETDEVICECAPABILITIESREQUEST._serialized_end=5793
  _SETDEVICECAPABILITIESRESPONSE._serialized_start=5795
  _SETDEVICECAPABILITIESRESPONSE._serialized_end=5826
  _GETTAGSREQUEST._serialized_start=5828
  _GETTAGSREQUEST._serialized_end=5904
  _GETTAGSRESPONSE._serialized_start=5907
  _GETTAGSRESPONSE._serialized_end=6111
  _GETTAGSRESPONSE_TAGSENTRY._serialized_start=541
  _GETTAGSRESPONSE_TAGSENTRY._serialized_end=640
  _UPSERTTAGREQUEST._serialized_start=6114
  _UPSERTTAGREQUEST._serialized_end=6254
  _UPSERTTAGRESPONSE._serialized_start=6256
  _UPSERTTAGRESPONSE._serialized_end=6275
  _DELETETAGREQUEST._serialized_start=6277
  _DELETETAGREQUEST._serialized_end=6373
  _DELETETAGRESPONSE._serialized_start=6375
  _DELETETAGRESPONSE._serialized_end=6394
  _FLEETMGRSERVICE._serialized_start=6677
  _FLEETMGRSERVICE._serialized_end=7809
  _FLEETMGREDGESERVICE._serialized_start=7812
  _FLEETMGREDGESERVICE._serialized_end=7958
  _FLEETMGRDEVICESERVICE._serialized_start=7961
  _FLEETMGRDEVICESERVICE._serialized_end=8829
  _FLEETMGRDEVICETAGSSERVICE._serialized_start=8832
  _FLEETMGRDEVICETAGSSERVICE._serialized_end=9798
# @@protoc_insertion_point(module_scope)
