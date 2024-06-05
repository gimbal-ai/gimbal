# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/api/corepb/v1/model_exec.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2
from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2
from src.common.typespb import uuid_pb2 as src_dot_common_dot_typespb_dot_uuid__pb2
from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\"src/api/corepb/v1/model_exec.proto\x12\x18gml.internal.api.core.v1\x1a\x14gogoproto/gogo.proto\x1a$mediapipe/framework/calculator.proto\x1a\x1dsrc/common/typespb/uuid.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"\xa8\x02\n\x04Node\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x45\n\x04kind\x18\x02 \x01(\x0e\x32\x31.gml.internal.api.core.v1.LogicalPipelineNodeKindR\x04kind\x12H\n\nattributes\x18\x03 \x03(\x0b\x32(.gml.internal.api.core.v1.NodeAttributesR\nattributes\x12;\n\x06inputs\x18\x04 \x03(\x0b\x32#.gml.internal.api.core.v1.NodeInputR\x06inputs\x12>\n\x07outputs\x18\x05 \x03(\x0b\x32$.gml.internal.api.core.v1.NodeOutputR\x07outputs\"\xbd\x03\n\x05Value\x12!\n\x0bstring_data\x18\x01 \x01(\tH\x00R\nstringData\x12\x1f\n\nint64_data\x18\x02 \x01(\x03H\x00R\tint64Data\x12!\n\x0b\x64ouble_data\x18\x03 \x01(\x01H\x00R\ndoubleData\x12\x1d\n\tbool_data\x18\x04 \x01(\x08H\x00R\x08\x62oolData\x12I\n\x0blambda_data\x18\x05 \x01(\x0b\x32&.gml.internal.api.core.v1.Value.LambdaH\x00R\nlambdaData\x12I\n\nmodel_data\x18\x06 \x01(\x0b\x32(.gml.internal.api.core.v1.Value.ModelRefH\x00R\tmodelData\x1a\x1e\n\x08ModelRef\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x1ap\n\x06Lambda\x12\x16\n\x06inputs\x18\x01 \x03(\tR\x06inputs\x12\x18\n\x07outputs\x18\x02 \x03(\tR\x07outputs\x12\x34\n\x05nodes\x18\x03 \x03(\x0b\x32\x1e.gml.internal.api.core.v1.NodeR\x05nodesB\x06\n\x04\x64\x61ta\"[\n\x0eNodeAttributes\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x35\n\x05value\x18\x02 \x01(\x0b\x32\x1f.gml.internal.api.core.v1.ValueR\x05value\"\xbc\x02\n\tNodeInput\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12O\n\x0bparam_value\x18\x02 \x01(\x0b\x32,.gml.internal.api.core.v1.NodeInput.ParamRefH\x00R\nparamValue\x12_\n\x11node_output_value\x18\x03 \x01(\x0b\x32\x31.gml.internal.api.core.v1.NodeInput.NodeOutputRefH\x00R\x0fnodeOutputValue\x1a\x1e\n\x08ParamRef\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x1a@\n\rNodeOutputRef\x12\x1b\n\tnode_name\x18\x01 \x01(\x03R\x08nodeName\x12\x12\n\x04name\x18\x02 \x01(\tR\x04nameB\x07\n\x05value\" \n\nNodeOutput\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\"a\n\x05Param\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x44\n\rdefault_value\x18\x02 \x01(\x0b\x32\x1f.gml.internal.api.core.v1.ValueR\x0c\x64\x65\x66\x61ultValue\"\x80\x01\n\x0fLogicalPipeline\x12\x37\n\x06params\x18\x01 \x03(\x0b\x32\x1f.gml.internal.api.core.v1.ParamR\x06params\x12\x34\n\x05nodes\x18\x02 \x03(\x0b\x32\x1e.gml.internal.api.core.v1.NodeR\x05nodes\"\xf0\x03\n\x12PipelineDeployment\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id\x12V\n\x13logical_pipeline_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x15\xe2\xde\x1f\x11LogicalPipelineIDR\x11logicalPipelineId\x12\x37\n\x08\x66leet_id\x18\x03 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0b\xe2\xde\x1f\x07\x46leetIDR\x07\x66leetId\x12\x39\n\ncreated_at\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tcreatedAt\x12\x39\n\nupdated_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tupdatedAt\x12\x18\n\x07version\x18\x06 \x01(\x03R\x07version\x12\x44\n\x04spec\x18\x07 \x01(\x0b\x32\x30.gml.internal.api.core.v1.PipelineDeploymentSpecR\x04spec\x12J\n\x06status\x18\x08 \x01(\x0b\x32\x32.gml.internal.api.core.v1.PipelineDeploymentStatusR\x06status\"W\n\x16PipelineDeploymentSpec\x12=\n\x05state\x18\x01 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\"Y\n\x18PipelineDeploymentStatus\x12=\n\x05state\x18\x01 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\"\xf6\x03\n\x10PhysicalPipeline\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id\x12_\n\x16pipeline_deployment_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x18\xe2\xde\x1f\x14PipelineDeploymentIDR\x14pipelineDeploymentId\x12:\n\tdevice_id\x18\x03 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\x12\x39\n\ncreated_at\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tcreatedAt\x12\x39\n\nupdated_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tupdatedAt\x12\x18\n\x07version\x18\x06 \x01(\x03R\x07version\x12\x42\n\x04spec\x18\x07 \x01(\x0b\x32..gml.internal.api.core.v1.PhysicalPipelineSpecR\x04spec\x12H\n\x06status\x18\x08 \x01(\x0b\x32\x30.gml.internal.api.core.v1.PhysicalPipelineStatusR\x06status\"U\n\x14PhysicalPipelineSpec\x12=\n\x05state\x18\x01 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\"W\n\x16PhysicalPipelineStatus\x12=\n\x05state\x18\x01 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\"H\n\x08Pipeline\x12<\n\x05nodes\x18\x02 \x03(\x0b\x32&.gml.internal.api.core.v1.PipelineNodeR\x05nodes\"\xa3\x02\n\x0cPipelineNode\x12\x0e\n\x02id\x18\x01 \x01(\x04R\x02id\x12\x12\n\x04type\x18\x02 \x01(\tR\x04type\x12\x36\n\x06inputs\x18\x03 \x03(\x0b\x32\x1e.gml.internal.api.core.v1.PortR\x06inputs\x12\x38\n\x07outputs\x18\x04 \x03(\x0b\x32\x1e.gml.internal.api.core.v1.PortR\x07outputs\x12\x44\n\x04\x61ttr\x18\x05 \x03(\x0b\x32\x30.gml.internal.api.core.v1.PipelineNode.AttrEntryR\x04\x61ttr\x1a\x37\n\tAttrEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\",\n\x04Port\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x10\n\x03net\x18\x02 \x01(\tR\x03net\"\x84\x01\n\x0c\x46ileResource\x12\x34\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\x12\x1d\n\nsize_bytes\x18\x02 \x01(\x04R\tsizeBytes\x12\x1f\n\x0bsha256_hash\x18\x03 \x01(\tR\nsha256Hash\"\x8b\x01\n\rExecutionSpec\x12\x36\n\x05graph\x18\x01 \x01(\x0b\x32 .mediapipe.CalculatorGraphConfigR\x05graph\x12\x42\n\nmodel_spec\x18\x02 \x03(\x0b\x32#.gml.internal.api.core.v1.ModelSpecR\tmodelSpec\"\x89\x03\n\tModelSpec\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x33\n\ronnx_blob_key\x18\x02 \x01(\tB\x0f\xe2\xde\x1f\x0bONNXBlobKeyR\x0bonnxBlobKey\x12Q\n\tonnx_file\x18\x03 \x01(\x0b\x32&.gml.internal.api.core.v1.FileResourceB\x0c\xe2\xde\x1f\x08ONNXFileR\x08onnxFile\x12\x18\n\x07runtime\x18\x32 \x01(\tR\x07runtime\x12\x62\n\rtensorrt_spec\x18\x64 \x01(\x0b\x32+.gml.internal.api.core.v1.TensorRTModelSpecB\x10\xe2\xde\x1f\x0cTensorRTSpecR\x0ctensorrtSpec\x12\x62\n\ropenvino_spec\x18\x65 \x01(\x0b\x32+.gml.internal.api.core.v1.OpenVINOModelSpecB\x10\xe2\xde\x1f\x0cOpenVINOSpecR\x0copenvinoSpec\"\xfe\x01\n\x11TensorRTModelSpec\x12h\n\x14optimization_profile\x18\x01 \x03(\x0b\x32\x35.gml.internal.api.core.v1.TensorRTOptimizationProfileR\x13optimizationProfile\x12&\n\x0f\x65ngine_blob_key\x18\x02 \x01(\tR\rengineBlobKey\x12W\n\x0fmem_pool_limits\x18\x03 \x01(\x0b\x32/.gml.internal.api.core.v1.TensorRTMemPoolLimitsR\rmemPoolLimits\"\x7f\n\x1bTensorRTOptimizationProfile\x12`\n\x12tensor_shape_range\x18\x01 \x03(\x0b\x32\x32.gml.internal.api.core.v1.TensorRTTensorShapeRangeR\x10tensorShapeRange\"M\n\x18TensorRTTensorShapeRange\x12\x1f\n\x0btensor_name\x18\x01 \x01(\tR\ntensorName\x12\x10\n\x03\x64im\x18\x02 \x03(\x05R\x03\x64im\"5\n\x15TensorRTMemPoolLimits\x12\x1c\n\tworkspace\x18\x01 \x01(\x03R\tworkspace\"\x8e\x01\n\x11OpenVINOModelSpec\x12X\n\x0binput_shape\x18\x01 \x03(\x0b\x32\x37.gml.internal.api.core.v1.OpenVINOModelSpec.TensorShapeR\ninputShape\x1a\x1f\n\x0bTensorShape\x12\x10\n\x03\x64im\x18\x01 \x03(\x05R\x03\x64im*\xc5\x03\n\x17LogicalPipelineNodeKind\x12&\n\"LOGICAL_PIPELINE_NODE_KIND_UNKNOWN\x10\x00\x12,\n(LOGICAL_PIPELINE_NODE_KIND_CAMERA_SOURCE\x10\n\x12&\n!LOGICAL_PIPELINE_NODE_KIND_DETECT\x10\xe8\x07\x12(\n#LOGICAL_PIPELINE_NODE_KIND_CLASSIFY\x10\xe9\x07\x12,\n\'LOGICAL_PIPELINE_NODE_KIND_FOR_EACH_ROI\x10\xdf\x0b\x12\x31\n,LOGICAL_PIPELINE_NODE_KIND_VIDEO_STREAM_SINK\x10\xd0\x0f\x12\x37\n2LOGICAL_PIPELINE_NODE_KIND_DETECTIONS_METRICS_SINK\x10\xd1\x0f\x12\x34\n/LOGICAL_PIPELINE_NODE_KIND_LATENCY_METRICS_SINK\x10\xd2\x0f\x12\x32\n-LOGICAL_PIPELINE_NODE_KIND_FRAME_METRICS_SINK\x10\xd3\x0f*\xd7\x01\n\rPipelineState\x12\x1a\n\x16PIPELINE_STATE_UNKNOWN\x10\x00\x12\x1a\n\x16PIPELINE_STATE_PENDING\x10\x01\x12\x18\n\x14PIPELINE_STATE_READY\x10\x02\x12\x1a\n\x16PIPELINE_STATE_RUNNING\x10\x03\x12\x1e\n\x1aPIPELINE_STATE_TERMINATING\x10\x04\x12\x1d\n\x19PIPELINE_STATE_TERMINATED\x10\x05\x12\x19\n\x15PIPELINE_STATE_FAILED\x10\x06\x42/Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.api.corepb.v1.model_exec_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepb'
  _PIPELINEDEPLOYMENT.fields_by_name['id']._options = None
  _PIPELINEDEPLOYMENT.fields_by_name['id']._serialized_options = b'\342\336\037\002ID'
  _PIPELINEDEPLOYMENT.fields_by_name['logical_pipeline_id']._options = None
  _PIPELINEDEPLOYMENT.fields_by_name['logical_pipeline_id']._serialized_options = b'\342\336\037\021LogicalPipelineID'
  _PIPELINEDEPLOYMENT.fields_by_name['fleet_id']._options = None
  _PIPELINEDEPLOYMENT.fields_by_name['fleet_id']._serialized_options = b'\342\336\037\007FleetID'
  _PHYSICALPIPELINE.fields_by_name['id']._options = None
  _PHYSICALPIPELINE.fields_by_name['id']._serialized_options = b'\342\336\037\002ID'
  _PHYSICALPIPELINE.fields_by_name['pipeline_deployment_id']._options = None
  _PHYSICALPIPELINE.fields_by_name['pipeline_deployment_id']._serialized_options = b'\342\336\037\024PipelineDeploymentID'
  _PHYSICALPIPELINE.fields_by_name['device_id']._options = None
  _PHYSICALPIPELINE.fields_by_name['device_id']._serialized_options = b'\342\336\037\010DeviceID'
  _PIPELINENODE_ATTRENTRY._options = None
  _PIPELINENODE_ATTRENTRY._serialized_options = b'8\001'
  _FILERESOURCE.fields_by_name['file_id']._options = None
  _FILERESOURCE.fields_by_name['file_id']._serialized_options = b'\342\336\037\006FileID'
  _MODELSPEC.fields_by_name['onnx_blob_key']._options = None
  _MODELSPEC.fields_by_name['onnx_blob_key']._serialized_options = b'\342\336\037\013ONNXBlobKey'
  _MODELSPEC.fields_by_name['onnx_file']._options = None
  _MODELSPEC.fields_by_name['onnx_file']._serialized_options = b'\342\336\037\010ONNXFile'
  _MODELSPEC.fields_by_name['tensorrt_spec']._options = None
  _MODELSPEC.fields_by_name['tensorrt_spec']._serialized_options = b'\342\336\037\014TensorRTSpec'
  _MODELSPEC.fields_by_name['openvino_spec']._options = None
  _MODELSPEC.fields_by_name['openvino_spec']._serialized_options = b'\342\336\037\014OpenVINOSpec'
  _LOGICALPIPELINENODEKIND._serialized_start=4724
  _LOGICALPIPELINENODEKIND._serialized_end=5177
  _PIPELINESTATE._serialized_start=5180
  _PIPELINESTATE._serialized_end=5395
  _NODE._serialized_start=189
  _NODE._serialized_end=485
  _VALUE._serialized_start=488
  _VALUE._serialized_end=933
  _VALUE_MODELREF._serialized_start=781
  _VALUE_MODELREF._serialized_end=811
  _VALUE_LAMBDA._serialized_start=813
  _VALUE_LAMBDA._serialized_end=925
  _NODEATTRIBUTES._serialized_start=935
  _NODEATTRIBUTES._serialized_end=1026
  _NODEINPUT._serialized_start=1029
  _NODEINPUT._serialized_end=1345
  _NODEINPUT_PARAMREF._serialized_start=1240
  _NODEINPUT_PARAMREF._serialized_end=1270
  _NODEINPUT_NODEOUTPUTREF._serialized_start=1272
  _NODEINPUT_NODEOUTPUTREF._serialized_end=1336
  _NODEOUTPUT._serialized_start=1347
  _NODEOUTPUT._serialized_end=1379
  _PARAM._serialized_start=1381
  _PARAM._serialized_end=1478
  _LOGICALPIPELINE._serialized_start=1481
  _LOGICALPIPELINE._serialized_end=1609
  _PIPELINEDEPLOYMENT._serialized_start=1612
  _PIPELINEDEPLOYMENT._serialized_end=2108
  _PIPELINEDEPLOYMENTSPEC._serialized_start=2110
  _PIPELINEDEPLOYMENTSPEC._serialized_end=2197
  _PIPELINEDEPLOYMENTSTATUS._serialized_start=2199
  _PIPELINEDEPLOYMENTSTATUS._serialized_end=2288
  _PHYSICALPIPELINE._serialized_start=2291
  _PHYSICALPIPELINE._serialized_end=2793
  _PHYSICALPIPELINESPEC._serialized_start=2795
  _PHYSICALPIPELINESPEC._serialized_end=2880
  _PHYSICALPIPELINESTATUS._serialized_start=2882
  _PHYSICALPIPELINESTATUS._serialized_end=2969
  _PIPELINE._serialized_start=2971
  _PIPELINE._serialized_end=3043
  _PIPELINENODE._serialized_start=3046
  _PIPELINENODE._serialized_end=3337
  _PIPELINENODE_ATTRENTRY._serialized_start=3282
  _PIPELINENODE_ATTRENTRY._serialized_end=3337
  _PORT._serialized_start=3339
  _PORT._serialized_end=3383
  _FILERESOURCE._serialized_start=3386
  _FILERESOURCE._serialized_end=3518
  _EXECUTIONSPEC._serialized_start=3521
  _EXECUTIONSPEC._serialized_end=3660
  _MODELSPEC._serialized_start=3663
  _MODELSPEC._serialized_end=4056
  _TENSORRTMODELSPEC._serialized_start=4059
  _TENSORRTMODELSPEC._serialized_end=4313
  _TENSORRTOPTIMIZATIONPROFILE._serialized_start=4315
  _TENSORRTOPTIMIZATIONPROFILE._serialized_end=4442
  _TENSORRTTENSORSHAPERANGE._serialized_start=4444
  _TENSORRTTENSORSHAPERANGE._serialized_end=4521
  _TENSORRTMEMPOOLLIMITS._serialized_start=4523
  _TENSORRTMEMPOOLLIMITS._serialized_end=4576
  _OPENVINOMODELSPEC._serialized_start=4579
  _OPENVINOMODELSPEC._serialized_end=4721
  _OPENVINOMODELSPEC_TENSORSHAPE._serialized_start=4690
  _OPENVINOMODELSPEC_TENSORSHAPE._serialized_end=4721
# @@protoc_insertion_point(module_scope)
