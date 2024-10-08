# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/util/tracking/frame_selection.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.util.tracking import camera_motion_pb2 as mediapipe_dot_util_dot_tracking_dot_camera__motion__pb2
from mediapipe.util.tracking import frame_selection_solution_evaluator_pb2 as mediapipe_dot_util_dot_tracking_dot_frame__selection__solution__evaluator__pb2
from mediapipe.util.tracking import region_flow_pb2 as mediapipe_dot_util_dot_tracking_dot_region__flow__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n-mediapipe/util/tracking/frame_selection.proto\x12\tmediapipe\x1a+mediapipe/util/tracking/camera_motion.proto\x1a@mediapipe/util/tracking/frame_selection_solution_evaluator.proto\x1a)mediapipe/util/tracking/region_flow.proto\"\x92\x01\n\x17\x46rameSelectionTimestamp\x12\x1c\n\ttimestamp\x18\x01 \x01(\x03R\ttimestamp\x12\x1b\n\tframe_idx\x18\x02 \x01(\x05R\x08\x66rameIdx\x12<\n\x18processed_from_timestamp\x18\x03 \x01(\x03:\x02-1R\x16processedFromTimestamp\"\x8b\x02\n\x14\x46rameSelectionResult\x12\x1c\n\ttimestamp\x18\x01 \x01(\x03R\ttimestamp\x12\x1b\n\tframe_idx\x18\x02 \x01(\x05R\x08\x66rameIdx\x12<\n\rcamera_motion\x18\x03 \x01(\x0b\x32\x17.mediapipe.CameraMotionR\x0c\x63\x61meraMotion\x12<\n\x08\x66\x65\x61tures\x18\x04 \x01(\x0b\x32 .mediapipe.RegionFlowFeatureListR\x08\x66\x65\x61tures\x12<\n\x18processed_from_timestamp\x18\x05 \x01(\x03:\x02-1R\x16processedFromTimestamp\"\xb3\x02\n\x17\x46rameSelectionCriterion\x12&\n\rsampling_rate\x18\x01 \x01(\x05:\x01\x30R\x0csamplingRate\x12-\n\x10\x62\x61ndwidth_frames\x18\x02 \x01(\x02:\x02\x35\x30R\x0f\x62\x61ndwidthFrames\x12\x33\n\x14search_radius_frames\x18\x03 \x01(\x05:\x01\x31R\x12searchRadiusFrames\x12]\n\x12solution_evaluator\x18\x04 \x01(\x0b\x32..mediapipe.FrameSelectionSolutionEvaluatorTypeR\x11solutionEvaluator\x12-\n\x11max_output_frames\x18\x05 \x01(\x05:\x01\x30R\x0fmaxOutputFrames\"}\n\x15\x46rameSelectionOptions\x12@\n\tcriterion\x18\x01 \x03(\x0b\x32\".mediapipe.FrameSelectionCriterionR\tcriterion\x12\"\n\nchunk_size\x18\x02 \x01(\x05:\x03\x31\x30\x30R\tchunkSizeB5Z3github.com/google/mediapipe/mediapipe/util/tracking')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.util.tracking.frame_selection_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z3github.com/google/mediapipe/mediapipe/util/tracking'
  _FRAMESELECTIONTIMESTAMP._serialized_start=215
  _FRAMESELECTIONTIMESTAMP._serialized_end=361
  _FRAMESELECTIONRESULT._serialized_start=364
  _FRAMESELECTIONRESULT._serialized_end=631
  _FRAMESELECTIONCRITERION._serialized_start=634
  _FRAMESELECTIONCRITERION._serialized_end=941
  _FRAMESELECTIONOPTIONS._serialized_start=943
  _FRAMESELECTIONOPTIONS._serialized_end=1068
# @@protoc_insertion_point(module_scope)
