# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/api/corepb/v1/cp_edge.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2
from src.common.typespb import uuid_pb2 as src_dot_common_dot_typespb_dot_uuid__pb2
from src.common.typespb import status_pb2 as src_dot_common_dot_typespb_dot_status__pb2
from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2
from opentelemetry.proto.metrics.v1 import metrics_pb2 as opentelemetry_dot_proto_dot_metrics_dot_v1_dot_metrics__pb2
from src.api.corepb.v1 import model_exec_pb2 as src_dot_api_dot_corepb_dot_v1_dot_model__exec__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1fsrc/api/corepb/v1/cp_edge.proto\x12\x18gml.internal.api.core.v1\x1a\x14gogoproto/gogo.proto\x1a\x1dsrc/common/typespb/uuid.proto\x1a\x1fsrc/common/typespb/status.proto\x1a\x19google/protobuf/any.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a,opentelemetry/proto/metrics/v1/metrics.proto\x1a\"src/api/corepb/v1/model_exec.proto\"1\n\rEdgeHeartbeat\x12 \n\x06seq_id\x18\x01 \x01(\x03\x42\t\xe2\xde\x1f\x05SeqIDR\x05seqId\"4\n\x10\x45\x64geHeartbeatAck\x12 \n\x06seq_id\x18\x01 \x01(\x03\x42\t\xe2\xde\x1f\x05SeqIDR\x05seqId\"\xbb\x01\n\x1aPhysicalPipelineSpecUpdate\x12Y\n\x14physical_pipeline_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x16\xe2\xde\x1f\x12PhysicalPipelineIDR\x12physicalPipelineId\x12\x42\n\x04spec\x18\x02 \x01(\x0b\x32..gml.internal.api.core.v1.PhysicalPipelineSpecR\x04spec\"\xdd\x01\n\x1cPhysicalPipelineStatusUpdate\x12Y\n\x14physical_pipeline_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x16\xe2\xde\x1f\x12PhysicalPipelineIDR\x12physicalPipelineId\x12\x18\n\x07version\x18\x02 \x01(\x03R\x07version\x12H\n\x06status\x18\x03 \x01(\x0b\x32\x30.gml.internal.api.core.v1.PhysicalPipelineStatusR\x06status\"\x0c\n\nCPRunModel\"\x0f\n\rCPRunModelAck\"\xb2\x01\n\x12\x45xecutionGraphSpec\x12=\n\x05graph\x18\x01 \x01(\x0b\x32\'.gml.internal.api.core.v1.ExecutionSpecR\x05graph\x12\x43\n\x05state\x18\x02 \x01(\x0e\x32-.gml.internal.api.core.v1.ExecutionGraphStateR\x05state\x12\x18\n\x07version\x18\x03 \x01(\x03R\x07version\"\x8d\x01\n\x14\x45xecutionGraphStatus\x12\x43\n\x05state\x18\x01 \x01(\x0e\x32-.gml.internal.api.core.v1.ExecutionGraphStateR\x05state\x12\x16\n\x06reason\x18\x02 \x01(\tR\x06reason\x12\x18\n\x07version\x18\x03 \x01(\x03R\x07version\"\x8a\x02\n\x13\x41pplyExecutionGraph\x12Y\n\x14physical_pipeline_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x16\xe2\xde\x1f\x12PhysicalPipelineIDR\x12physicalPipelineId\x12V\n\x13logical_pipeline_id\x18\x03 \x01(\x0b\x32\x0f.gml.types.UUIDB\x15\xe2\xde\x1f\x11LogicalPipelineIDR\x11logicalPipelineId\x12@\n\x04spec\x18\x02 \x01(\x0b\x32,.gml.internal.api.core.v1.ExecutionGraphSpecR\x04spec\"q\n\x14\x44\x65leteExecutionGraph\x12Y\n\x14physical_pipeline_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x16\xe2\xde\x1f\x12PhysicalPipelineIDR\x12physicalPipelineId\"\xbf\x01\n\x1a\x45xecutionGraphStatusUpdate\x12Y\n\x14physical_pipeline_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x16\xe2\xde\x1f\x12PhysicalPipelineIDR\x12physicalPipelineId\x12\x46\n\x06status\x18\x02 \x01(\x0b\x32..gml.internal.api.core.v1.ExecutionGraphStatusR\x06status\"\x12\n\x10VideoStreamStart\"\x11\n\x0fVideoStreamStop\"\x16\n\x14VideoStreamKeepAlive\"m\n\x0f\x45\x64geOTelMetrics\x12Z\n\x10resource_metrics\x18\x01 \x01(\x0b\x32/.opentelemetry.proto.metrics.v1.ResourceMetricsR\x0fresourceMetrics\"\x94\x01\n\x13\x46ileTransferRequest\x12\x34\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\x12*\n\x11\x63hunk_start_bytes\x18\x02 \x01(\x03R\x0f\x63hunkStartBytes\x12\x1b\n\tnum_bytes\x18\x03 \x01(\x03R\x08numBytes\"\x8f\x02\n\x14\x46ileTransferResponse\x12)\n\x06status\x18\x01 \x01(\x0b\x32\x11.gml.types.StatusR\x06status\x12N\n\x05\x63hunk\x18\x02 \x01(\x0b\x32\x38.gml.internal.api.core.v1.FileTransferResponse.FileChunkR\x05\x63hunk\x12\x34\n\x07\x66ile_id\x18\x03 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\x1a\x46\n\tFileChunk\x12\x1f\n\x0bstart_bytes\x18\x01 \x01(\x03R\nstartBytes\x12\x18\n\x07payload\x18\x02 \x01(\x0cR\x07payload\"\xb0\x05\n\x12\x44\x65viceCapabilities\x12\x64\n\x0emodel_runtimes\x18\x01 \x03(\x0b\x32=.gml.internal.api.core.v1.DeviceCapabilities.ModelRuntimeInfoR\rmodelRuntimes\x12Q\n\x07\x63\x61meras\x18\x02 \x03(\x0b\x32\x37.gml.internal.api.core.v1.DeviceCapabilities.CameraInfoR\x07\x63\x61meras\x1a\xec\x01\n\x10ModelRuntimeInfo\x12\x62\n\x04type\x18\x01 \x01(\x0e\x32N.gml.internal.api.core.v1.DeviceCapabilities.ModelRuntimeInfo.ModelRuntimeTypeR\x04type\"t\n\x10ModelRuntimeType\x12\x1e\n\x1aMODEL_RUNTIME_TYPE_UNKNOWN\x10\x00\x12\x1f\n\x1bMODEL_RUNTIME_TYPE_TENSORRT\x10\x01\x12\x1f\n\x1bMODEL_RUNTIME_TYPE_OPENVINO\x10\x02\x1a\xf1\x01\n\nCameraInfo\x12\\\n\x06\x64river\x18\x01 \x01(\x0e\x32\x44.gml.internal.api.core.v1.DeviceCapabilities.CameraInfo.CameraDriverR\x06\x64river\x12)\n\tcamera_id\x18\x02 \x01(\tB\x0c\xe2\xde\x1f\x08\x43\x61meraIDR\x08\x63\x61meraId\"Z\n\x0c\x43\x61meraDriver\x12\x19\n\x15\x43\x41MERA_DRIVER_UNKNOWN\x10\x00\x12\x17\n\x13\x43\x41MERA_DRIVER_ARGUS\x10\x01\x12\x16\n\x12\x43\x41MERA_DRIVER_V4L2\x10\x02\"\xcc\x01\n\x0e\x45\x64geCPMetadata\x12;\n\x05topic\x18\x01 \x01(\x0e\x32%.gml.internal.api.core.v1.EdgeCPTopicR\x05topic\x12:\n\tdevice_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\x12\x41\n\x0erecv_timestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\rrecvTimestamp\"~\n\rEdgeCPMessage\x12\x44\n\x08metadata\x18\x01 \x01(\x0b\x32(.gml.internal.api.core.v1.EdgeCPMetadataR\x08metadata\x12\'\n\x03msg\x18\xe8\x07 \x01(\x0b\x32\x14.google.protobuf.AnyR\x03msg\"\xcc\x01\n\x0e\x43PEdgeMetadata\x12;\n\x05topic\x18\x01 \x01(\x0e\x32%.gml.internal.api.core.v1.CPEdgeTopicR\x05topic\x12:\n\tdevice_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\x12\x41\n\x0erecv_timestamp\x18\x03 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\rrecvTimestamp\"~\n\rCPEdgeMessage\x12\x44\n\x08metadata\x18\x01 \x01(\x0b\x32(.gml.internal.api.core.v1.CPEdgeMetadataR\x08metadata\x12\'\n\x03msg\x18\xe8\x07 \x01(\x0b\x32\x14.google.protobuf.AnyR\x03msg*\xbe\x02\n\x13\x45xecutionGraphState\x12!\n\x1d\x45XECUTION_GRAPH_STATE_UNKNOWN\x10\x00\x12*\n&EXECUTION_GRAPH_STATE_UPDATE_REQUESTED\x10\n\x12%\n!EXECUTION_GRAPH_STATE_DOWNLOADING\x10\x14\x12#\n\x1f\x45XECUTION_GRAPH_STATE_COMPILING\x10\x1e\x12\x1f\n\x1b\x45XECUTION_GRAPH_STATE_READY\x10(\x12\"\n\x1e\x45XECUTION_GRAPH_STATE_DEPLOYED\x10\x32\x12%\n!EXECUTION_GRAPH_STATE_TERMINATING\x10<\x12 \n\x1c\x45XECUTION_GRAPH_STATE_FAILED\x10\x64*\xc7\x01\n\x0b\x45\x64geCPTopic\x12\x19\n\x15\x45\x44GE_CP_TOPIC_UNKNOWN\x10\x00\x12\x18\n\x14\x45\x44GE_CP_TOPIC_STATUS\x10\x01\x12\x17\n\x13\x45\x44GE_CP_TOPIC_VIDEO\x10\x02\x12\x16\n\x12\x45\x44GE_CP_TOPIC_EXEC\x10\x03\x12\x19\n\x15\x45\x44GE_CP_TOPIC_METRICS\x10\x04\x12\x1f\n\x1b\x45\x44GE_CP_TOPIC_FILE_TRANSFER\x10\x05\x12\x16\n\x12\x45\x44GE_CP_TOPIC_INFO\x10\x06*\xc7\x01\n\x0b\x43PEdgeTopic\x12\x19\n\x15\x43P_EDGE_TOPIC_UNKNOWN\x10\x00\x12\x18\n\x14\x43P_EDGE_TOPIC_STATUS\x10\x01\x12\x17\n\x13\x43P_EDGE_TOPIC_VIDEO\x10\x02\x12\x16\n\x12\x43P_EDGE_TOPIC_EXEC\x10\x03\x12\x19\n\x15\x43P_EDGE_TOPIC_METRICS\x10\x04\x12\x1f\n\x1b\x43P_EDGE_TOPIC_FILE_TRANSFER\x10\x05\x12\x16\n\x12\x43P_EDGE_TOPIC_INFO\x10\x06\x42/Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.api.corepb.v1.cp_edge_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepb'
  _EDGEHEARTBEAT.fields_by_name['seq_id']._options = None
  _EDGEHEARTBEAT.fields_by_name['seq_id']._serialized_options = b'\342\336\037\005SeqID'
  _EDGEHEARTBEATACK.fields_by_name['seq_id']._options = None
  _EDGEHEARTBEATACK.fields_by_name['seq_id']._serialized_options = b'\342\336\037\005SeqID'
  _PHYSICALPIPELINESPECUPDATE.fields_by_name['physical_pipeline_id']._options = None
  _PHYSICALPIPELINESPECUPDATE.fields_by_name['physical_pipeline_id']._serialized_options = b'\342\336\037\022PhysicalPipelineID'
  _PHYSICALPIPELINESTATUSUPDATE.fields_by_name['physical_pipeline_id']._options = None
  _PHYSICALPIPELINESTATUSUPDATE.fields_by_name['physical_pipeline_id']._serialized_options = b'\342\336\037\022PhysicalPipelineID'
  _APPLYEXECUTIONGRAPH.fields_by_name['physical_pipeline_id']._options = None
  _APPLYEXECUTIONGRAPH.fields_by_name['physical_pipeline_id']._serialized_options = b'\342\336\037\022PhysicalPipelineID'
  _APPLYEXECUTIONGRAPH.fields_by_name['logical_pipeline_id']._options = None
  _APPLYEXECUTIONGRAPH.fields_by_name['logical_pipeline_id']._serialized_options = b'\342\336\037\021LogicalPipelineID'
  _DELETEEXECUTIONGRAPH.fields_by_name['physical_pipeline_id']._options = None
  _DELETEEXECUTIONGRAPH.fields_by_name['physical_pipeline_id']._serialized_options = b'\342\336\037\022PhysicalPipelineID'
  _EXECUTIONGRAPHSTATUSUPDATE.fields_by_name['physical_pipeline_id']._options = None
  _EXECUTIONGRAPHSTATUSUPDATE.fields_by_name['physical_pipeline_id']._serialized_options = b'\342\336\037\022PhysicalPipelineID'
  _FILETRANSFERREQUEST.fields_by_name['file_id']._options = None
  _FILETRANSFERREQUEST.fields_by_name['file_id']._serialized_options = b'\342\336\037\006FileID'
  _FILETRANSFERRESPONSE.fields_by_name['file_id']._options = None
  _FILETRANSFERRESPONSE.fields_by_name['file_id']._serialized_options = b'\342\336\037\006FileID'
  _DEVICECAPABILITIES_CAMERAINFO.fields_by_name['camera_id']._options = None
  _DEVICECAPABILITIES_CAMERAINFO.fields_by_name['camera_id']._serialized_options = b'\342\336\037\010CameraID'
  _EDGECPMETADATA.fields_by_name['device_id']._options = None
  _EDGECPMETADATA.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _CPEDGEMETADATA.fields_by_name['device_id']._options = None
  _CPEDGEMETADATA.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _EXECUTIONGRAPHSTATE._serialized_start=3703
  _EXECUTIONGRAPHSTATE._serialized_end=4021
  _EDGECPTOPIC._serialized_start=4024
  _EDGECPTOPIC._serialized_end=4223
  _CPEDGETOPIC._serialized_start=4226
  _CPEDGETOPIC._serialized_end=4425
  _EDGEHEARTBEAT._serialized_start=289
  _EDGEHEARTBEAT._serialized_end=338
  _EDGEHEARTBEATACK._serialized_start=340
  _EDGEHEARTBEATACK._serialized_end=392
  _PHYSICALPIPELINESPECUPDATE._serialized_start=395
  _PHYSICALPIPELINESPECUPDATE._serialized_end=582
  _PHYSICALPIPELINESTATUSUPDATE._serialized_start=585
  _PHYSICALPIPELINESTATUSUPDATE._serialized_end=806
  _CPRUNMODEL._serialized_start=808
  _CPRUNMODEL._serialized_end=820
  _CPRUNMODELACK._serialized_start=822
  _CPRUNMODELACK._serialized_end=837
  _EXECUTIONGRAPHSPEC._serialized_start=840
  _EXECUTIONGRAPHSPEC._serialized_end=1018
  _EXECUTIONGRAPHSTATUS._serialized_start=1021
  _EXECUTIONGRAPHSTATUS._serialized_end=1162
  _APPLYEXECUTIONGRAPH._serialized_start=1165
  _APPLYEXECUTIONGRAPH._serialized_end=1431
  _DELETEEXECUTIONGRAPH._serialized_start=1433
  _DELETEEXECUTIONGRAPH._serialized_end=1546
  _EXECUTIONGRAPHSTATUSUPDATE._serialized_start=1549
  _EXECUTIONGRAPHSTATUSUPDATE._serialized_end=1740
  _VIDEOSTREAMSTART._serialized_start=1742
  _VIDEOSTREAMSTART._serialized_end=1760
  _VIDEOSTREAMSTOP._serialized_start=1762
  _VIDEOSTREAMSTOP._serialized_end=1779
  _VIDEOSTREAMKEEPALIVE._serialized_start=1781
  _VIDEOSTREAMKEEPALIVE._serialized_end=1803
  _EDGEOTELMETRICS._serialized_start=1805
  _EDGEOTELMETRICS._serialized_end=1914
  _FILETRANSFERREQUEST._serialized_start=1917
  _FILETRANSFERREQUEST._serialized_end=2065
  _FILETRANSFERRESPONSE._serialized_start=2068
  _FILETRANSFERRESPONSE._serialized_end=2339
  _FILETRANSFERRESPONSE_FILECHUNK._serialized_start=2269
  _FILETRANSFERRESPONSE_FILECHUNK._serialized_end=2339
  _DEVICECAPABILITIES._serialized_start=2342
  _DEVICECAPABILITIES._serialized_end=3030
  _DEVICECAPABILITIES_MODELRUNTIMEINFO._serialized_start=2550
  _DEVICECAPABILITIES_MODELRUNTIMEINFO._serialized_end=2786
  _DEVICECAPABILITIES_MODELRUNTIMEINFO_MODELRUNTIMETYPE._serialized_start=2670
  _DEVICECAPABILITIES_MODELRUNTIMEINFO_MODELRUNTIMETYPE._serialized_end=2786
  _DEVICECAPABILITIES_CAMERAINFO._serialized_start=2789
  _DEVICECAPABILITIES_CAMERAINFO._serialized_end=3030
  _DEVICECAPABILITIES_CAMERAINFO_CAMERADRIVER._serialized_start=2940
  _DEVICECAPABILITIES_CAMERAINFO_CAMERADRIVER._serialized_end=3030
  _EDGECPMETADATA._serialized_start=3033
  _EDGECPMETADATA._serialized_end=3237
  _EDGECPMESSAGE._serialized_start=3239
  _EDGECPMESSAGE._serialized_end=3365
  _CPEDGEMETADATA._serialized_start=3368
  _CPEDGEMETADATA._serialized_end=3572
  _CPEDGEMESSAGE._serialized_start=3574
  _CPEDGEMESSAGE._serialized_end=3700
# @@protoc_insertion_point(module_scope)
