# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/detections_to_rects_calculator.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import calculator_pb2 as mediapipe_dot_framework_dot_calculator__pb2
try:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe_dot_framework_dot_calculator__options__pb2
except AttributeError:
  mediapipe_dot_framework_dot_calculator__options__pb2 = mediapipe_dot_framework_dot_calculator__pb2.mediapipe.framework.calculator_options_pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n?mediapipe/calculators/util/detections_to_rects_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\xb3\x05\n\"DetectionsToRectsCalculatorOptions\x12N\n$rotation_vector_start_keypoint_index\x18\x01 \x01(\x05R rotationVectorStartKeypointIndex\x12J\n\"rotation_vector_end_keypoint_index\x18\x02 \x01(\x05R\x1erotationVectorEndKeypointIndex\x12?\n\x1crotation_vector_target_angle\x18\x03 \x01(\x02R\x19rotationVectorTargetAngle\x12N\n$rotation_vector_target_angle_degrees\x18\x04 \x01(\x02R rotationVectorTargetAngleDegrees\x12O\n%output_zero_rect_for_empty_detections\x18\x05 \x01(\x08R outputZeroRectForEmptyDetections\x12\x65\n\x0f\x63onversion_mode\x18\x06 \x01(\x0e\x32<.mediapipe.DetectionsToRectsCalculatorOptions.ConversionModeR\x0e\x63onversionMode\"F\n\x0e\x43onversionMode\x12\x0b\n\x07\x44\x45\x46\x41ULT\x10\x00\x12\x14\n\x10USE_BOUNDING_BOX\x10\x01\x12\x11\n\rUSE_KEYPOINTS\x10\x02\x32`\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xdf\xb7\xa1} \x01(\x0b\x32-.mediapipe.DetectionsToRectsCalculatorOptionsR\x03\x65xtB8Z6github.com/google/mediapipe/mediapipe/calculators/util')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.detections_to_rects_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_DETECTIONSTORECTSCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z6github.com/google/mediapipe/mediapipe/calculators/util'
  _DETECTIONSTORECTSCALCULATOROPTIONS._serialized_start=117
  _DETECTIONSTORECTSCALCULATOROPTIONS._serialized_end=808
  _DETECTIONSTORECTSCALCULATOROPTIONS_CONVERSIONMODE._serialized_start=640
  _DETECTIONSTORECTSCALCULATOROPTIONS_CONVERSIONMODE._serialized_end=710
# @@protoc_insertion_point(module_scope)