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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\"src/api/corepb/v1/model_exec.proto\x12\x18gml.internal.api.core.v1\x1a\x14gogoproto/gogo.proto\x1a$mediapipe/framework/calculator.proto\x1a\x1dsrc/common/typespb/uuid.proto\x1a\x1fgoogle/protobuf/timestamp.proto\"\xa3\x02\n\x04Node\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x45\n\x04kind\x18\x02 \x01(\x0e\x32\x31.gml.internal.api.core.v1.LogicalPipelineNodeKindR\x04kind\x12;\n\x06inputs\x18\x03 \x03(\x0b\x32#.gml.internal.api.core.v1.NodeInputR\x06inputs\x12>\n\x07outputs\x18\x04 \x03(\x0b\x32$.gml.internal.api.core.v1.NodeOutputR\x07outputs\x12\x43\n\tinit_args\x18\x05 \x03(\x0b\x32&.gml.internal.api.core.v1.NodeInitArgsR\x08initArgs\"\xc1\x01\n\x0cNodeInitArgs\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12#\n\x0cstring_value\x18\x02 \x01(\tH\x00R\x0bstringValue\x12!\n\x0bint64_value\x18\x03 \x01(\x03H\x00R\nint64Value\x12#\n\x0c\x64ouble_value\x18\x04 \x01(\x01H\x00R\x0b\x64oubleValue\x12\x1f\n\nbool_value\x18\x05 \x01(\x08H\x00R\tboolValueB\x0f\n\rdefault_value\"\xcb\x03\n\tNodeInput\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12Q\n\x0bparam_value\x18\x02 \x01(\x0b\x32..gml.internal.api.core.v1.NodeInput.ParamInputH\x00R\nparamValue\x12Q\n\x0bmodel_value\x18\x03 \x01(\x0b\x32..gml.internal.api.core.v1.NodeInput.ModelInputH\x00R\nmodelValue\x12_\n\x11node_output_value\x18\x04 \x01(\x0b\x32\x31.gml.internal.api.core.v1.NodeInput.NodeOutputRefH\x00R\x0fnodeOutputValue\x1a@\n\rNodeOutputRef\x12\x1b\n\tnode_name\x18\x01 \x01(\x03R\x08nodeName\x12\x12\n\x04name\x18\x02 \x01(\tR\x04name\x1a+\n\nModelInput\x12\x1d\n\nmodel_name\x18\x01 \x01(\tR\tmodelName\x1a+\n\nParamInput\x12\x1d\n\nparam_name\x18\x01 \x01(\tR\tparamNameB\x07\n\x05value\" \n\nNodeOutput\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\"\xc0\x01\n\x0bGlobalParam\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12#\n\x0cstring_value\x18\x02 \x01(\tH\x00R\x0bstringValue\x12!\n\x0bint64_value\x18\x03 \x01(\x03H\x00R\nint64Value\x12#\n\x0c\x64ouble_value\x18\x04 \x01(\x01H\x00R\x0b\x64oubleValue\x12\x1f\n\nbool_value\x18\x05 \x01(\x08H\x00R\tboolValueB\x0f\n\rdefault_value\"\x93\x01\n\x0fLogicalPipeline\x12J\n\rglobal_params\x18\x01 \x03(\x0b\x32%.gml.internal.api.core.v1.GlobalParamR\x0cglobalParams\x12\x34\n\x05nodes\x18\x02 \x03(\x0b\x32\x1e.gml.internal.api.core.v1.NodeR\x05nodes\"\xd2\x03\n\x12PipelineDeployment\x12\x1f\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDR\x02id\x12?\n\x13logical_pipeline_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDR\x11logicalPipelineId\x12*\n\x08\x66leet_id\x18\x03 \x01(\x0b\x32\x0f.gml.types.UUIDR\x07\x66leetId\x12\x39\n\ncreated_at\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tcreatedAt\x12\x39\n\nupdated_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tupdatedAt\x12)\n\x10resource_version\x18\x06 \x01(\x03R\x0fresourceVersion\x12\x44\n\x04spec\x18\x07 \x01(\x0b\x32\x30.gml.internal.api.core.v1.PipelineDeploymentSpecR\x04spec\x12G\n\x05state\x18\x08 \x01(\x0b\x32\x31.gml.internal.api.core.v1.PipelineDeploymentStateR\x05state\"W\n\x16PipelineDeploymentSpec\x12=\n\x05state\x18\x01 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\"X\n\x17PipelineDeploymentState\x12=\n\x05state\x18\x02 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\"H\n\x08Pipeline\x12<\n\x05nodes\x18\x02 \x03(\x0b\x32&.gml.internal.api.core.v1.PipelineNodeR\x05nodes\"\xa3\x02\n\x0cPipelineNode\x12\x0e\n\x02id\x18\x01 \x01(\x04R\x02id\x12\x12\n\x04type\x18\x02 \x01(\tR\x04type\x12\x36\n\x06inputs\x18\x03 \x03(\x0b\x32\x1e.gml.internal.api.core.v1.PortR\x06inputs\x12\x38\n\x07outputs\x18\x04 \x03(\x0b\x32\x1e.gml.internal.api.core.v1.PortR\x07outputs\x12\x44\n\x04\x61ttr\x18\x05 \x03(\x0b\x32\x30.gml.internal.api.core.v1.PipelineNode.AttrEntryR\x04\x61ttr\x1a\x37\n\tAttrEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12\x14\n\x05value\x18\x02 \x01(\tR\x05value:\x02\x38\x01\",\n\x04Port\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x10\n\x03net\x18\x02 \x01(\tR\x03net\"\x84\x01\n\x0c\x46ileResource\x12\x34\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\x12\x1d\n\nsize_bytes\x18\x02 \x01(\x04R\tsizeBytes\x12\x1f\n\x0bsha256_hash\x18\x03 \x01(\tR\nsha256Hash\"\x8b\x01\n\rExecutionSpec\x12\x36\n\x05graph\x18\x01 \x01(\x0b\x32 .mediapipe.CalculatorGraphConfigR\x05graph\x12\x42\n\nmodel_spec\x18\x02 \x03(\x0b\x32#.gml.internal.api.core.v1.ModelSpecR\tmodelSpec\"\x89\x03\n\tModelSpec\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x33\n\ronnx_blob_key\x18\x02 \x01(\tB\x0f\xe2\xde\x1f\x0bONNXBlobKeyR\x0bonnxBlobKey\x12Q\n\tonnx_file\x18\x03 \x01(\x0b\x32&.gml.internal.api.core.v1.FileResourceB\x0c\xe2\xde\x1f\x08ONNXFileR\x08onnxFile\x12\x18\n\x07runtime\x18\x32 \x01(\tR\x07runtime\x12\x62\n\rtensorrt_spec\x18\x64 \x01(\x0b\x32+.gml.internal.api.core.v1.TensorRTModelSpecB\x10\xe2\xde\x1f\x0cTensorRTSpecR\x0ctensorrtSpec\x12\x62\n\ropenvino_spec\x18\x65 \x01(\x0b\x32+.gml.internal.api.core.v1.OpenVINOModelSpecB\x10\xe2\xde\x1f\x0cOpenVINOSpecR\x0copenvinoSpec\"\xfe\x01\n\x11TensorRTModelSpec\x12h\n\x14optimization_profile\x18\x01 \x03(\x0b\x32\x35.gml.internal.api.core.v1.TensorRTOptimizationProfileR\x13optimizationProfile\x12&\n\x0f\x65ngine_blob_key\x18\x02 \x01(\tR\rengineBlobKey\x12W\n\x0fmem_pool_limits\x18\x03 \x01(\x0b\x32/.gml.internal.api.core.v1.TensorRTMemPoolLimitsR\rmemPoolLimits\"\x7f\n\x1bTensorRTOptimizationProfile\x12`\n\x12tensor_shape_range\x18\x01 \x03(\x0b\x32\x32.gml.internal.api.core.v1.TensorRTTensorShapeRangeR\x10tensorShapeRange\"M\n\x18TensorRTTensorShapeRange\x12\x1f\n\x0btensor_name\x18\x01 \x01(\tR\ntensorName\x12\x10\n\x03\x64im\x18\x02 \x03(\x05R\x03\x64im\"5\n\x15TensorRTMemPoolLimits\x12\x1c\n\tworkspace\x18\x01 \x01(\x03R\tworkspace\"\x8e\x01\n\x11OpenVINOModelSpec\x12X\n\x0binput_shape\x18\x01 \x03(\x0b\x32\x37.gml.internal.api.core.v1.OpenVINOModelSpec.TensorShapeR\ninputShape\x1a\x1f\n\x0bTensorShape\x12\x10\n\x03\x64im\x18\x01 \x03(\x05R\x03\x64im*\x8c\x02\n\x17LogicalPipelineNodeKind\x12&\n\"LOGICAL_PIPELINE_NODE_KIND_UNKNOWN\x10\x00\x12,\n(LOGICAL_PIPELINE_NODE_KIND_CAMERA_SOURCE\x10\x01\x12/\n*LOGICAL_PIPELINE_NODE_KIND_DETECTION_MODEL\x10\xe8\x07\x12\x31\n,LOGICAL_PIPELINE_NODE_KIND_VIDEO_STREAM_SINK\x10\xd0\x0f\x12\x37\n2LOGICAL_PIPELINE_NODE_KIND_DETECTIONS_METRICS_SINK\x10\xd1\x0f*\xd7\x01\n\rPipelineState\x12\x1a\n\x16PIPELINE_STATE_UNKNOWN\x10\x00\x12\x1a\n\x16PIPELINE_STATE_PENDING\x10\x01\x12\x18\n\x14PIPELINE_STATE_READY\x10\x02\x12\x1a\n\x16PIPELINE_STATE_RUNNING\x10\x03\x12\x1e\n\x1aPIPELINE_STATE_TERMINATING\x10\x04\x12\x1d\n\x19PIPELINE_STATE_TERMINATED\x10\x05\x12\x19\n\x15PIPELINE_STATE_FAILED\x10\x06\x42/Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.api.corepb.v1.model_exec_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepb'
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
  _LOGICALPIPELINENODEKIND._serialized_start=3920
  _LOGICALPIPELINENODEKIND._serialized_end=4188
  _PIPELINESTATE._serialized_start=4191
  _PIPELINESTATE._serialized_end=4406
  _NODE._serialized_start=189
  _NODE._serialized_end=480
  _NODEINITARGS._serialized_start=483
  _NODEINITARGS._serialized_end=676
  _NODEINPUT._serialized_start=679
  _NODEINPUT._serialized_end=1138
  _NODEINPUT_NODEOUTPUTREF._serialized_start=975
  _NODEINPUT_NODEOUTPUTREF._serialized_end=1039
  _NODEINPUT_MODELINPUT._serialized_start=1041
  _NODEINPUT_MODELINPUT._serialized_end=1084
  _NODEINPUT_PARAMINPUT._serialized_start=1086
  _NODEINPUT_PARAMINPUT._serialized_end=1129
  _NODEOUTPUT._serialized_start=1140
  _NODEOUTPUT._serialized_end=1172
  _GLOBALPARAM._serialized_start=1175
  _GLOBALPARAM._serialized_end=1367
  _LOGICALPIPELINE._serialized_start=1370
  _LOGICALPIPELINE._serialized_end=1517
  _PIPELINEDEPLOYMENT._serialized_start=1520
  _PIPELINEDEPLOYMENT._serialized_end=1986
  _PIPELINEDEPLOYMENTSPEC._serialized_start=1988
  _PIPELINEDEPLOYMENTSPEC._serialized_end=2075
  _PIPELINEDEPLOYMENTSTATE._serialized_start=2077
  _PIPELINEDEPLOYMENTSTATE._serialized_end=2165
  _PIPELINE._serialized_start=2167
  _PIPELINE._serialized_end=2239
  _PIPELINENODE._serialized_start=2242
  _PIPELINENODE._serialized_end=2533
  _PIPELINENODE_ATTRENTRY._serialized_start=2478
  _PIPELINENODE_ATTRENTRY._serialized_end=2533
  _PORT._serialized_start=2535
  _PORT._serialized_end=2579
  _FILERESOURCE._serialized_start=2582
  _FILERESOURCE._serialized_end=2714
  _EXECUTIONSPEC._serialized_start=2717
  _EXECUTIONSPEC._serialized_end=2856
  _MODELSPEC._serialized_start=2859
  _MODELSPEC._serialized_end=3252
  _TENSORRTMODELSPEC._serialized_start=3255
  _TENSORRTMODELSPEC._serialized_end=3509
  _TENSORRTOPTIMIZATIONPROFILE._serialized_start=3511
  _TENSORRTOPTIMIZATIONPROFILE._serialized_end=3638
  _TENSORRTTENSORSHAPERANGE._serialized_start=3640
  _TENSORRTTENSORSHAPERANGE._serialized_end=3717
  _TENSORRTMEMPOOLLIMITS._serialized_start=3719
  _TENSORRTMEMPOOLLIMITS._serialized_end=3772
  _OPENVINOMODELSPEC._serialized_start=3775
  _OPENVINOMODELSPEC._serialized_end=3917
  _OPENVINOMODELSPEC_TENSORSHAPE._serialized_start=3886
  _OPENVINOMODELSPEC_TENSORSHAPE._serialized_end=3917
# @@protoc_insertion_point(module_scope)
