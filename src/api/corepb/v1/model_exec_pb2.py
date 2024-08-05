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
from google.protobuf import wrappers_pb2 as google_dot_protobuf_dot_wrappers__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\"src/api/corepb/v1/model_exec.proto\x12\x18gml.internal.api.core.v1\x1a\x14gogoproto/gogo.proto\x1a$mediapipe/framework/calculator.proto\x1a\x1dsrc/common/typespb/uuid.proto\x1a\x1fgoogle/protobuf/timestamp.proto\x1a\x1egoogle/protobuf/wrappers.proto\"\xa7\x02\n\x0f\x42oundingBoxInfo\x12Z\n\nbox_format\x18\x01 \x01(\x0e\x32;.gml.internal.api.core.v1.BoundingBoxInfo.BoundingBoxFormatR\tboxFormat\x12%\n\x0e\x62ox_normalized\x18\x02 \x01(\x08R\rboxNormalized\"\x90\x01\n\x11\x42oundingBoxFormat\x12\x1f\n\x1b\x42OUNDING_BOX_FORMAT_UNKNOWN\x10\x00\x12\x1e\n\x1a\x42OUNDING_BOX_FORMAT_CXCYWH\x10\x01\x12\x1c\n\x18\x42OUNDING_BOX_FORMAT_YXYX\x10\x02\x12\x1c\n\x18\x42OUNDING_BOX_FORMAT_XYXY\x10\x03\"\xf4\x07\n\x16ImagePreprocessingStep\x12[\n\x04kind\x18\x01 \x01(\x0e\x32G.gml.internal.api.core.v1.ImagePreprocessingStep.ImagePreprocessingKindR\x04kind\x12s\n\x11\x63onversion_params\x18\x02 \x01(\x0b\x32\x46.gml.internal.api.core.v1.ImagePreprocessingStep.ImageConversionParamsR\x10\x63onversionParams\x12g\n\rresize_params\x18\x03 \x01(\x0b\x32\x42.gml.internal.api.core.v1.ImagePreprocessingStep.ImageResizeParamsR\x0cresizeParams\x12v\n\x12standardize_params\x18\x04 \x01(\x0b\x32G.gml.internal.api.core.v1.ImagePreprocessingStep.ImageStandardizeParamsR\x11standardizeParams\x1a-\n\x15ImageConversionParams\x12\x14\n\x05scale\x18\x01 \x01(\x08R\x05scale\x1a\xed\x01\n\x11ImageResizeParams\x12\x66\n\x04kind\x18\x01 \x01(\x0e\x32R.gml.internal.api.core.v1.ImagePreprocessingStep.ImageResizeParams.ImageResizeKindR\x04kind\"p\n\x0fImageResizeKind\x12\x1d\n\x19IMAGE_RESIZE_KIND_UNKNOWN\x10\x00\x12\x1d\n\x19IMAGE_RESIZE_KIND_STRETCH\x10\x01\x12\x1f\n\x1bIMAGE_RESIZE_KIND_LETTERBOX\x10\x02\x1aH\n\x16ImageStandardizeParams\x12\x14\n\x05means\x18\x01 \x03(\x02R\x05means\x12\x18\n\x07stddevs\x18\x02 \x03(\x02R\x07stddevs\"\xbd\x01\n\x16ImagePreprocessingKind\x12$\n IMAGE_PREPROCESSING_KIND_UNKNOWN\x10\x00\x12.\n*IMAGE_PREPROCESSING_KIND_CONVERT_TO_TENSOR\x10\x01\x12#\n\x1fIMAGE_PREPROCESSING_KIND_RESIZE\x10\x02\x12(\n$IMAGE_PREPROCESSING_KIND_STANDARDIZE\x10\x03\"\x9e\x14\n\x12\x44imensionSemantics\x12W\n\x04kind\x18\x01 \x01(\x0e\x32\x43.gml.internal.api.core.v1.DimensionSemantics.DimensionSemanticsKindR\x04kind\x12q\n\x14image_channel_params\x18\x02 \x01(\x0b\x32?.gml.internal.api.core.v1.DimensionSemantics.ImageChannelParamsR\x12imageChannelParams\x12\x86\x01\n\x1b\x64\x65tection_candidates_params\x18\x03 \x01(\x0b\x32\x46.gml.internal.api.core.v1.DimensionSemantics.DetectionCandidatesParamsR\x19\x64\x65tectionCandidatesParams\x12z\n\x17\x64\x65tection_output_params\x18\x04 \x01(\x0b\x32\x42.gml.internal.api.core.v1.DimensionSemantics.DetectionOutputParamsR\x15\x64\x65tectionOutputParams\x12}\n\x18segmentation_mask_params\x18\x05 \x01(\x0b\x32\x43.gml.internal.api.core.v1.DimensionSemantics.SegmentationMaskParamsR\x16segmentationMaskParams\x12j\n\x11regression_params\x18\x06 \x01(\x0b\x32=.gml.internal.api.core.v1.DimensionSemantics.RegressionParamsR\x10regressionParams\x1a\xf4\x01\n\x12ImageChannelParams\x12j\n\x06\x66ormat\x18\x01 \x01(\x0e\x32R.gml.internal.api.core.v1.DimensionSemantics.ImageChannelParams.ImageChannelFormatR\x06\x66ormat\"r\n\x12ImageChannelFormat\x12 \n\x1cIMAGE_CHANNEL_FORMAT_UNKNOWN\x10\x00\x12\x1c\n\x18IMAGE_CHANNEL_FORMAT_RGB\x10\x01\x12\x1c\n\x18IMAGE_CHANNEL_FORMAT_BGR\x10\x02\x1a=\n\x19\x44\x65tectionCandidatesParams\x12 \n\x0cis_nms_boxes\x18\x01 \x01(\x08R\nisNmsBoxes\x1a\xdf\x03\n\x15\x44\x65tectionOutputParams\x12\x7f\n\x14\x62ox_coordinate_range\x18\x01 \x01(\x0b\x32M.gml.internal.api.core.v1.DimensionSemantics.DetectionOutputParams.IndexRangeR\x12\x62oxCoordinateRange\x12H\n\nbox_format\x18\x02 \x01(\x0b\x32).gml.internal.api.core.v1.BoundingBoxInfoR\tboxFormat\x12\x30\n\x14\x62ox_confidence_index\x18\x03 \x01(\x05R\x12\x62oxConfidenceIndex\x12\x1f\n\x0b\x63lass_index\x18\x04 \x01(\x05R\nclassIndex\x12p\n\x0cscores_range\x18\x05 \x01(\x0b\x32M.gml.internal.api.core.v1.DimensionSemantics.DetectionOutputParams.IndexRangeR\x0bscoresRange\x1a\x36\n\nIndexRange\x12\x14\n\x05start\x18\x01 \x01(\x05R\x05start\x12\x12\n\x04size\x18\x02 \x01(\x05R\x04size\x1a\xae\x02\n\x16SegmentationMaskParams\x12l\n\x04kind\x18\x01 \x01(\x0e\x32X.gml.internal.api.core.v1.DimensionSemantics.SegmentationMaskParams.SegmentationMaskKindR\x04kind\"\xa5\x01\n\x14SegmentationMaskKind\x12\"\n\x1eSEGMENTATION_MASK_KIND_UNKNOWN\x10\x00\x12&\n\"SEGMENTATION_MASK_KIND_CLASS_LABEL\x10\x01\x12\x1f\n\x1bSEGMENTATION_MASK_KIND_BOOL\x10\x02\x12 \n\x1cSEGMENTATION_MASK_KIND_SCORE\x10\x03\x1a\\\n\x10RegressionParams\x12\x14\n\x05label\x18\x01 \x01(\tR\x05label\x12\x32\n\x05scale\x18\x02 \x01(\x0b\x32\x1c.google.protobuf.DoubleValueR\x05scale\"\xa4\x05\n\x16\x44imensionSemanticsKind\x12$\n DIMENSION_SEMANTICS_KIND_UNKNOWN\x10\x00\x12\"\n\x1e\x44IMENSION_SEMANTICS_KIND_BATCH\x10\x01\x12#\n\x1f\x44IMENSION_SEMANTICS_KIND_IGNORE\x10\x02\x12*\n&DIMENSION_SEMANTICS_KIND_IMAGE_CHANNEL\x10\x03\x12)\n%DIMENSION_SEMANTICS_KIND_IMAGE_HEIGHT\x10\x04\x12(\n$DIMENSION_SEMANTICS_KIND_IMAGE_WIDTH\x10\x05\x12\x31\n-DIMENSION_SEMANTICS_KIND_DETECTION_CANDIDATES\x10\x06\x12-\n)DIMENSION_SEMANTICS_KIND_DETECTION_OUTPUT\x10\x07\x12)\n%DIMENSION_SEMANTICS_KIND_CLASS_SCORES\x10\x08\x12\x36\n2DIMENSION_SEMANTICS_KIND_SEGMENTATION_MASK_CHANNEL\x10\t\x12)\n%DIMENSION_SEMANTICS_KIND_CLASS_LABELS\x10\n\x12-\n)DIMENSION_SEMANTICS_KIND_REGRESSION_VALUE\x10\x0b\x12#\n\x1f\x44IMENSION_SEMANTICS_KIND_TOKENS\x10\x0c\x12+\n\'DIMENSION_SEMANTICS_KIND_ATTENTION_MASK\x10\r\x12)\n%DIMENSION_SEMANTICS_KIND_VOCAB_LOGITS\x10\x0e\"_\n\x0fTensorSemantics\x12L\n\ndimensions\x18\x01 \x03(\x0b\x32,.gml.internal.api.core.v1.DimensionSemanticsR\ndimensions\"\xa7\t\n\tModelInfo\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x41\n\x04kind\x18\x02 \x01(\x0e\x32-.gml.internal.api.core.v1.ModelInfo.ModelKindR\x04kind\x12N\n\x06\x66ormat\x18\x03 \x01(\x0e\x32\x36.gml.internal.api.core.v1.ModelInfo.ModelStorageFormatR\x06\x66ormat\x12T\n\x0b\x66ile_assets\x18\x04 \x03(\x0b\x32\x33.gml.internal.api.core.v1.ModelInfo.FileAssetsEntryR\nfileAssets\x12_\n\x16input_tensor_semantics\x18\x05 \x03(\x0b\x32).gml.internal.api.core.v1.TensorSemanticsR\x14inputTensorSemantics\x12\x61\n\x17output_tensor_semantics\x18\x06 \x03(\x0b\x32).gml.internal.api.core.v1.TensorSemanticsR\x15outputTensorSemantics\x12!\n\x0c\x63lass_labels\x18\x64 \x03(\tR\x0b\x63lassLabels\x12\x46\n\tbbox_info\x18\x65 \x01(\x0b\x32).gml.internal.api.core.v1.BoundingBoxInfoR\x08\x62\x62oxInfo\x12l\n\x19image_preprocessing_steps\x18\x66 \x03(\x0b\x32\x30.gml.internal.api.core.v1.ImagePreprocessingStepR\x17imagePreprocessingSteps\x1aN\n\x0f\x46ileAssetsEntry\x12\x10\n\x03key\x18\x01 \x01(\tR\x03key\x12%\n\x05value\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDR\x05value:\x02\x38\x01\"\x9a\x01\n\tModelKind\x12\x16\n\x12MODEL_KIND_UNKNOWN\x10\x00\x12\x14\n\x10MODEL_KIND_TORCH\x10\x01\x12\x1a\n\x16MODEL_KIND_TORCHSCRIPT\x10\x02\x12\x13\n\x0fMODEL_KIND_ONNX\x10\x03\x12\x15\n\x11MODEL_KIND_TFLITE\x10\x04\x12\x17\n\x13MODEL_KIND_OPENVINO\x10\x05\"\x92\x02\n\x12ModelStorageFormat\x12 \n\x1cMODEL_STORAGE_FORMAT_UNKNOWN\x10\x00\x12&\n\"MODEL_STORAGE_FORMAT_MLIR_BYTECODE\x10\x01\x12\"\n\x1eMODEL_STORAGE_FORMAT_MLIR_TEXT\x10\x02\x12!\n\x1dMODEL_STORAGE_FORMAT_PROTOBUF\x10\x03\x12#\n\x1fMODEL_STORAGE_FORMAT_PROTO_TEXT\x10\x04\x12#\n\x1fMODEL_STORAGE_FORMAT_FLATBUFFER\x10\x05\x12!\n\x1dMODEL_STORAGE_FORMAT_OPENVINO\x10\x06\"\xa8\x02\n\x04Node\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x45\n\x04kind\x18\x02 \x01(\x0e\x32\x31.gml.internal.api.core.v1.LogicalPipelineNodeKindR\x04kind\x12H\n\nattributes\x18\x03 \x03(\x0b\x32(.gml.internal.api.core.v1.NodeAttributesR\nattributes\x12;\n\x06inputs\x18\x04 \x03(\x0b\x32#.gml.internal.api.core.v1.NodeInputR\x06inputs\x12>\n\x07outputs\x18\x05 \x03(\x0b\x32$.gml.internal.api.core.v1.NodeOutputR\x07outputs\"\x1e\n\x08ParamRef\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\"\xab\x04\n\x05Value\x12!\n\x0bstring_data\x18\x01 \x01(\tH\x00R\nstringData\x12\x1f\n\nint64_data\x18\x02 \x01(\x03H\x00R\tint64Data\x12!\n\x0b\x64ouble_data\x18\x03 \x01(\x01H\x00R\ndoubleData\x12\x1d\n\tbool_data\x18\x04 \x01(\x08H\x00R\x08\x62oolData\x12I\n\x0blambda_data\x18\x05 \x01(\x0b\x32&.gml.internal.api.core.v1.Value.LambdaH\x00R\nlambdaData\x12I\n\nmodel_data\x18\x06 \x01(\x0b\x32(.gml.internal.api.core.v1.Value.ModelRefH\x00R\tmodelData\x12\x43\n\nparam_data\x18\x07 \x01(\x0b\x32\".gml.internal.api.core.v1.ParamRefH\x00R\tparamData\x1aG\n\x08ModelRef\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\'\n\x02id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id\x1ap\n\x06Lambda\x12\x16\n\x06inputs\x18\x01 \x03(\tR\x06inputs\x12\x18\n\x07outputs\x18\x02 \x03(\tR\x07outputs\x12\x34\n\x05nodes\x18\x03 \x03(\x0b\x32\x1e.gml.internal.api.core.v1.NodeR\x05nodesB\x06\n\x04\x64\x61ta\"[\n\x0eNodeAttributes\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x35\n\x05value\x18\x02 \x01(\x0b\x32\x1f.gml.internal.api.core.v1.ValueR\x05value\"\x9c\x03\n\tNodeInput\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x45\n\x0bparam_value\x18\x02 \x01(\x0b\x32\".gml.internal.api.core.v1.ParamRefH\x00R\nparamValue\x12_\n\x11node_output_value\x18\x03 \x01(\x0b\x32\x31.gml.internal.api.core.v1.NodeInput.NodeOutputRefH\x00R\x0fnodeOutputValue\x12\x62\n\x12lambda_input_value\x18\x04 \x01(\x0b\x32\x32.gml.internal.api.core.v1.NodeInput.LambdaInputRefH\x00R\x10lambdaInputValue\x1a@\n\rNodeOutputRef\x12\x1b\n\tnode_name\x18\x01 \x01(\tR\x08nodeName\x12\x12\n\x04name\x18\x02 \x01(\tR\x04name\x1a$\n\x0eLambdaInputRef\x12\x12\n\x04name\x18\x01 \x01(\tR\x04nameB\x07\n\x05value\" \n\nNodeOutput\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\"a\n\x05Param\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x44\n\rdefault_value\x18\x02 \x01(\x0b\x32\x1f.gml.internal.api.core.v1.ValueR\x0c\x64\x65\x66\x61ultValue\"\xbc\x01\n\x0fLogicalPipeline\x12\x37\n\x06params\x18\x01 \x03(\x0b\x32\x1f.gml.internal.api.core.v1.ParamR\x06params\x12\x34\n\x05nodes\x18\x02 \x03(\x0b\x32\x1e.gml.internal.api.core.v1.NodeR\x05nodes\x12:\n\tmodel_ids\x18\x03 \x03(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08ModelIDsR\x08modelIds\"\xab\x04\n\x12PipelineDeployment\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id\x12V\n\x13logical_pipeline_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x15\xe2\xde\x1f\x11LogicalPipelineIDR\x11logicalPipelineId\x12\x37\n\x08\x66leet_id\x18\x03 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0b\xe2\xde\x1f\x07\x46leetIDR\x07\x66leetId\x12\x39\n\ncreated_at\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tcreatedAt\x12\x39\n\nupdated_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tupdatedAt\x12\x18\n\x07version\x18\x06 \x01(\x03R\x07version\x12\x44\n\x04spec\x18\x07 \x01(\x0b\x32\x30.gml.internal.api.core.v1.PipelineDeploymentSpecR\x04spec\x12J\n\x06status\x18\x08 \x01(\x0b\x32\x32.gml.internal.api.core.v1.PipelineDeploymentStatusR\x06status\x12\x39\n\ndeleted_at\x18\t \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tdeletedAt\"W\n\x16PipelineDeploymentSpec\x12=\n\x05state\x18\x01 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\"q\n\x18PipelineDeploymentStatus\x12=\n\x05state\x18\x01 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\x12\x16\n\x06reason\x18\x02 \x01(\tR\x06reason\"\xdc\x03\n\x10PhysicalPipeline\x12\'\n\x02id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x06\xe2\xde\x1f\x02IDR\x02id\x12_\n\x16pipeline_deployment_id\x18\x02 \x01(\x0b\x32\x0f.gml.types.UUIDB\x18\xe2\xde\x1f\x14PipelineDeploymentIDR\x14pipelineDeploymentId\x12:\n\tdevice_id\x18\x03 \x01(\x0b\x32\x0f.gml.types.UUIDB\x0c\xe2\xde\x1f\x08\x44\x65viceIDR\x08\x64\x65viceId\x12\x39\n\ncreated_at\x18\x04 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tcreatedAt\x12\x39\n\nupdated_at\x18\x05 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\tupdatedAt\x12\x42\n\x04spec\x18\x06 \x01(\x0b\x32..gml.internal.api.core.v1.PhysicalPipelineSpecR\x04spec\x12H\n\x06status\x18\x07 \x01(\x0b\x32\x30.gml.internal.api.core.v1.PhysicalPipelineStatusR\x06status\"\xc8\x01\n\x14PhysicalPipelineSpec\x12=\n\x05state\x18\x01 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\x12\x18\n\x07version\x18\x02 \x01(\x03R\x07version\x12=\n\x05graph\x18\x03 \x01(\x0b\x32\'.gml.internal.api.core.v1.ExecutionSpecR\x05graph\x12\x18\n\x07runtime\x18\x04 \x01(\tR\x07runtime\"\xa3\x01\n\x16PhysicalPipelineStatus\x12=\n\x05state\x18\x01 \x01(\x0e\x32\'.gml.internal.api.core.v1.PipelineStateR\x05state\x12\x18\n\x07version\x18\x02 \x01(\x03R\x07version\x12\x18\n\x07runtime\x18\x03 \x01(\tR\x07runtime\x12\x16\n\x06reason\x18\x04 \x01(\tR\x06reason\"\x84\x01\n\x0c\x46ileResource\x12\x34\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\x12\x1d\n\nsize_bytes\x18\x02 \x01(\x04R\tsizeBytes\x12\x1f\n\x0bsha256_hash\x18\x03 \x01(\tR\nsha256Hash\"\x8b\x01\n\rExecutionSpec\x12\x36\n\x05graph\x18\x01 \x01(\x0b\x32 .mediapipe.CalculatorGraphConfigR\x05graph\x12\x42\n\nmodel_spec\x18\x02 \x03(\x0b\x32#.gml.internal.api.core.v1.ModelSpecR\tmodelSpec\"\\\n\nNamedAsset\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12:\n\x04\x66ile\x18\x02 \x01(\x0b\x32&.gml.internal.api.core.v1.FileResourceR\x04\x66ile\"\xd0\x03\n\tModelSpec\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12\x33\n\ronnx_blob_key\x18\x02 \x01(\tB\x0f\xe2\xde\x1f\x0bONNXBlobKeyR\x0bonnxBlobKey\x12Q\n\tonnx_file\x18\x03 \x01(\x0b\x32&.gml.internal.api.core.v1.FileResourceB\x0c\xe2\xde\x1f\x08ONNXFileR\x08onnxFile\x12\x45\n\x0bnamed_asset\x18\x04 \x03(\x0b\x32$.gml.internal.api.core.v1.NamedAssetR\nnamedAsset\x12\x18\n\x07runtime\x18\x32 \x01(\tR\x07runtime\x12\x62\n\rtensorrt_spec\x18\x64 \x01(\x0b\x32+.gml.internal.api.core.v1.TensorRTModelSpecB\x10\xe2\xde\x1f\x0cTensorRTSpecR\x0ctensorrtSpec\x12\x62\n\ropenvino_spec\x18\x65 \x01(\x0b\x32+.gml.internal.api.core.v1.OpenVINOModelSpecB\x10\xe2\xde\x1f\x0cOpenVINOSpecR\x0copenvinoSpec\"\xfe\x01\n\x11TensorRTModelSpec\x12h\n\x14optimization_profile\x18\x01 \x03(\x0b\x32\x35.gml.internal.api.core.v1.TensorRTOptimizationProfileR\x13optimizationProfile\x12&\n\x0f\x65ngine_blob_key\x18\x02 \x01(\tR\rengineBlobKey\x12W\n\x0fmem_pool_limits\x18\x03 \x01(\x0b\x32/.gml.internal.api.core.v1.TensorRTMemPoolLimitsR\rmemPoolLimits\"\x7f\n\x1bTensorRTOptimizationProfile\x12`\n\x12tensor_shape_range\x18\x01 \x03(\x0b\x32\x32.gml.internal.api.core.v1.TensorRTTensorShapeRangeR\x10tensorShapeRange\"M\n\x18TensorRTTensorShapeRange\x12\x1f\n\x0btensor_name\x18\x01 \x01(\tR\ntensorName\x12\x10\n\x03\x64im\x18\x02 \x03(\x05R\x03\x64im\"5\n\x15TensorRTMemPoolLimits\x12\x1c\n\tworkspace\x18\x01 \x01(\x03R\tworkspace\"\x8e\x01\n\x11OpenVINOModelSpec\x12X\n\x0binput_shape\x18\x01 \x03(\x0b\x32\x37.gml.internal.api.core.v1.OpenVINOModelSpec.TensorShapeR\ninputShape\x1a\x1f\n\x0bTensorShape\x12\x10\n\x03\x64im\x18\x01 \x03(\x05R\x03\x64im*\xd8\x05\n\x17LogicalPipelineNodeKind\x12&\n\"LOGICAL_PIPELINE_NODE_KIND_UNKNOWN\x10\x00\x12,\n(LOGICAL_PIPELINE_NODE_KIND_CAMERA_SOURCE\x10\n\x12\x31\n-LOGICAL_PIPELINE_NODE_KIND_TEXT_STREAM_SOURCE\x10\x0b\x12&\n!LOGICAL_PIPELINE_NODE_KIND_DETECT\x10\xe8\x07\x12(\n#LOGICAL_PIPELINE_NODE_KIND_CLASSIFY\x10\xe9\x07\x12\'\n\"LOGICAL_PIPELINE_NODE_KIND_SEGMENT\x10\xea\x07\x12\x33\n.LOGICAL_PIPELINE_NODE_KIND_MULTI_PURPOSE_MODEL\x10\xeb\x07\x12\'\n\"LOGICAL_PIPELINE_NODE_KIND_REGRESS\x10\xec\x07\x12%\n LOGICAL_PIPELINE_NODE_KIND_TRACK\x10\xed\x07\x12,\n\'LOGICAL_PIPELINE_NODE_KIND_FOR_EACH_ROI\x10\xdf\x0b\x12\x31\n,LOGICAL_PIPELINE_NODE_KIND_VIDEO_STREAM_SINK\x10\xd0\x0f\x12\x37\n2LOGICAL_PIPELINE_NODE_KIND_DETECTIONS_METRICS_SINK\x10\xd1\x0f\x12\x34\n/LOGICAL_PIPELINE_NODE_KIND_LATENCY_METRICS_SINK\x10\xd2\x0f\x12\x32\n-LOGICAL_PIPELINE_NODE_KIND_FRAME_METRICS_SINK\x10\xd3\x0f\x12\x30\n+LOGICAL_PIPELINE_NODE_KIND_TEXT_STREAM_SINK\x10\xd4\x0f*\xd7\x01\n\rPipelineState\x12\x1a\n\x16PIPELINE_STATE_UNKNOWN\x10\x00\x12\x1a\n\x16PIPELINE_STATE_PENDING\x10\x01\x12\x18\n\x14PIPELINE_STATE_READY\x10\x02\x12\x1a\n\x16PIPELINE_STATE_RUNNING\x10\x03\x12\x1e\n\x1aPIPELINE_STATE_TERMINATING\x10\x04\x12\x1d\n\x19PIPELINE_STATE_TERMINATED\x10\x05\x12\x19\n\x15PIPELINE_STATE_FAILED\x10\x06\x42/Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.api.corepb.v1.model_exec_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepb'
  _MODELINFO_FILEASSETSENTRY._options = None
  _MODELINFO_FILEASSETSENTRY._serialized_options = b'8\001'
  _VALUE_MODELREF.fields_by_name['id']._options = None
  _VALUE_MODELREF.fields_by_name['id']._serialized_options = b'\342\336\037\002ID'
  _LOGICALPIPELINE.fields_by_name['model_ids']._options = None
  _LOGICALPIPELINE.fields_by_name['model_ids']._serialized_options = b'\342\336\037\010ModelIDs'
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
  _LOGICALPIPELINENODEKIND._serialized_start=10252
  _LOGICALPIPELINENODEKIND._serialized_end=10980
  _PIPELINESTATE._serialized_start=10983
  _PIPELINESTATE._serialized_end=11198
  _BOUNDINGBOXINFO._serialized_start=221
  _BOUNDINGBOXINFO._serialized_end=516
  _BOUNDINGBOXINFO_BOUNDINGBOXFORMAT._serialized_start=372
  _BOUNDINGBOXINFO_BOUNDINGBOXFORMAT._serialized_end=516
  _IMAGEPREPROCESSINGSTEP._serialized_start=519
  _IMAGEPREPROCESSINGSTEP._serialized_end=1531
  _IMAGEPREPROCESSINGSTEP_IMAGECONVERSIONPARAMS._serialized_start=980
  _IMAGEPREPROCESSINGSTEP_IMAGECONVERSIONPARAMS._serialized_end=1025
  _IMAGEPREPROCESSINGSTEP_IMAGERESIZEPARAMS._serialized_start=1028
  _IMAGEPREPROCESSINGSTEP_IMAGERESIZEPARAMS._serialized_end=1265
  _IMAGEPREPROCESSINGSTEP_IMAGERESIZEPARAMS_IMAGERESIZEKIND._serialized_start=1153
  _IMAGEPREPROCESSINGSTEP_IMAGERESIZEPARAMS_IMAGERESIZEKIND._serialized_end=1265
  _IMAGEPREPROCESSINGSTEP_IMAGESTANDARDIZEPARAMS._serialized_start=1267
  _IMAGEPREPROCESSINGSTEP_IMAGESTANDARDIZEPARAMS._serialized_end=1339
  _IMAGEPREPROCESSINGSTEP_IMAGEPREPROCESSINGKIND._serialized_start=1342
  _IMAGEPREPROCESSINGSTEP_IMAGEPREPROCESSINGKIND._serialized_end=1531
  _DIMENSIONSEMANTICS._serialized_start=1534
  _DIMENSIONSEMANTICS._serialized_end=4124
  _DIMENSIONSEMANTICS_IMAGECHANNELPARAMS._serialized_start=2257
  _DIMENSIONSEMANTICS_IMAGECHANNELPARAMS._serialized_end=2501
  _DIMENSIONSEMANTICS_IMAGECHANNELPARAMS_IMAGECHANNELFORMAT._serialized_start=2387
  _DIMENSIONSEMANTICS_IMAGECHANNELPARAMS_IMAGECHANNELFORMAT._serialized_end=2501
  _DIMENSIONSEMANTICS_DETECTIONCANDIDATESPARAMS._serialized_start=2503
  _DIMENSIONSEMANTICS_DETECTIONCANDIDATESPARAMS._serialized_end=2564
  _DIMENSIONSEMANTICS_DETECTIONOUTPUTPARAMS._serialized_start=2567
  _DIMENSIONSEMANTICS_DETECTIONOUTPUTPARAMS._serialized_end=3046
  _DIMENSIONSEMANTICS_DETECTIONOUTPUTPARAMS_INDEXRANGE._serialized_start=2992
  _DIMENSIONSEMANTICS_DETECTIONOUTPUTPARAMS_INDEXRANGE._serialized_end=3046
  _DIMENSIONSEMANTICS_SEGMENTATIONMASKPARAMS._serialized_start=3049
  _DIMENSIONSEMANTICS_SEGMENTATIONMASKPARAMS._serialized_end=3351
  _DIMENSIONSEMANTICS_SEGMENTATIONMASKPARAMS_SEGMENTATIONMASKKIND._serialized_start=3186
  _DIMENSIONSEMANTICS_SEGMENTATIONMASKPARAMS_SEGMENTATIONMASKKIND._serialized_end=3351
  _DIMENSIONSEMANTICS_REGRESSIONPARAMS._serialized_start=3353
  _DIMENSIONSEMANTICS_REGRESSIONPARAMS._serialized_end=3445
  _DIMENSIONSEMANTICS_DIMENSIONSEMANTICSKIND._serialized_start=3448
  _DIMENSIONSEMANTICS_DIMENSIONSEMANTICSKIND._serialized_end=4124
  _TENSORSEMANTICS._serialized_start=4126
  _TENSORSEMANTICS._serialized_end=4221
  _MODELINFO._serialized_start=4224
  _MODELINFO._serialized_end=5415
  _MODELINFO_FILEASSETSENTRY._serialized_start=4903
  _MODELINFO_FILEASSETSENTRY._serialized_end=4981
  _MODELINFO_MODELKIND._serialized_start=4984
  _MODELINFO_MODELKIND._serialized_end=5138
  _MODELINFO_MODELSTORAGEFORMAT._serialized_start=5141
  _MODELINFO_MODELSTORAGEFORMAT._serialized_end=5415
  _NODE._serialized_start=5418
  _NODE._serialized_end=5714
  _PARAMREF._serialized_start=5716
  _PARAMREF._serialized_end=5746
  _VALUE._serialized_start=5749
  _VALUE._serialized_end=6304
  _VALUE_MODELREF._serialized_start=6111
  _VALUE_MODELREF._serialized_end=6182
  _VALUE_LAMBDA._serialized_start=6184
  _VALUE_LAMBDA._serialized_end=6296
  _NODEATTRIBUTES._serialized_start=6306
  _NODEATTRIBUTES._serialized_end=6397
  _NODEINPUT._serialized_start=6400
  _NODEINPUT._serialized_end=6812
  _NODEINPUT_NODEOUTPUTREF._serialized_start=6701
  _NODEINPUT_NODEOUTPUTREF._serialized_end=6765
  _NODEINPUT_LAMBDAINPUTREF._serialized_start=6767
  _NODEINPUT_LAMBDAINPUTREF._serialized_end=6803
  _NODEOUTPUT._serialized_start=6814
  _NODEOUTPUT._serialized_end=6846
  _PARAM._serialized_start=6848
  _PARAM._serialized_end=6945
  _LOGICALPIPELINE._serialized_start=6948
  _LOGICALPIPELINE._serialized_end=7136
  _PIPELINEDEPLOYMENT._serialized_start=7139
  _PIPELINEDEPLOYMENT._serialized_end=7694
  _PIPELINEDEPLOYMENTSPEC._serialized_start=7696
  _PIPELINEDEPLOYMENTSPEC._serialized_end=7783
  _PIPELINEDEPLOYMENTSTATUS._serialized_start=7785
  _PIPELINEDEPLOYMENTSTATUS._serialized_end=7898
  _PHYSICALPIPELINE._serialized_start=7901
  _PHYSICALPIPELINE._serialized_end=8377
  _PHYSICALPIPELINESPEC._serialized_start=8380
  _PHYSICALPIPELINESPEC._serialized_end=8580
  _PHYSICALPIPELINESTATUS._serialized_start=8583
  _PHYSICALPIPELINESTATUS._serialized_end=8746
  _FILERESOURCE._serialized_start=8749
  _FILERESOURCE._serialized_end=8881
  _EXECUTIONSPEC._serialized_start=8884
  _EXECUTIONSPEC._serialized_end=9023
  _NAMEDASSET._serialized_start=9025
  _NAMEDASSET._serialized_end=9117
  _MODELSPEC._serialized_start=9120
  _MODELSPEC._serialized_end=9584
  _TENSORRTMODELSPEC._serialized_start=9587
  _TENSORRTMODELSPEC._serialized_end=9841
  _TENSORRTOPTIMIZATIONPROFILE._serialized_start=9843
  _TENSORRTOPTIMIZATIONPROFILE._serialized_end=9970
  _TENSORRTTENSORSHAPERANGE._serialized_start=9972
  _TENSORRTTENSORSHAPERANGE._serialized_end=10049
  _TENSORRTMEMPOOLLIMITS._serialized_start=10051
  _TENSORRTMEMPOOLLIMITS._serialized_end=10104
  _OPENVINOMODELSPEC._serialized_start=10107
  _OPENVINOMODELSPEC._serialized_end=10249
  _OPENVINOMODELSPEC_TENSORSHAPE._serialized_start=10218
  _OPENVINOMODELSPEC_TENSORSHAPE._serialized_end=10249
# @@protoc_insertion_point(module_scope)
