# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/util/non_max_suppression_calculator.proto
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


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n?mediapipe/calculators/util/non_max_suppression_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\"\x97\x06\n\"NonMaxSuppressionCalculatorOptions\x12\x35\n\x15num_detection_streams\x18\x01 \x01(\x05:\x01\x31R\x13numDetectionStreams\x12\x30\n\x12max_num_detections\x18\x02 \x01(\x05:\x02-1R\x10maxNumDetections\x12\x32\n\x13min_score_threshold\x18\x06 \x01(\x02:\x02-1R\x11minScoreThreshold\x12=\n\x19min_suppression_threshold\x18\x03 \x01(\x02:\x01\x31R\x17minSuppressionThreshold\x12\x65\n\x0coverlap_type\x18\x04 \x01(\x0e\x32\x39.mediapipe.NonMaxSuppressionCalculatorOptions.OverlapType:\x07JACCARDR\x0boverlapType\x12\x36\n\x17return_empty_detections\x18\x05 \x01(\x08R\x15returnEmptyDetections\x12j\n\talgorithm\x18\x07 \x01(\x0e\x32:.mediapipe.NonMaxSuppressionCalculatorOptions.NmsAlgorithm:\x10NMS_ALGO_DEFAULTR\talgorithm\"k\n\x0bOverlapType\x12\x1c\n\x18UNSPECIFIED_OVERLAP_TYPE\x10\x00\x12\x0b\n\x07JACCARD\x10\x01\x12\x14\n\x10MODIFIED_JACCARD\x10\x02\x12\x1b\n\x17INTERSECTION_OVER_UNION\x10\x03\";\n\x0cNmsAlgorithm\x12\x14\n\x10NMS_ALGO_DEFAULT\x10\x00\x12\x15\n\x11NMS_ALGO_WEIGHTED\x10\x01\x32`\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xbc\xa8\xb4\x1a \x01(\x0b\x32-.mediapipe.NonMaxSuppressionCalculatorOptionsR\x03\x65xtB8Z6github.com/google/mediapipe/mediapipe/calculators/util')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.util.non_max_suppression_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_NONMAXSUPPRESSIONCALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z6github.com/google/mediapipe/mediapipe/calculators/util'
  _NONMAXSUPPRESSIONCALCULATOROPTIONS._serialized_start=117
  _NONMAXSUPPRESSIONCALCULATOROPTIONS._serialized_end=908
  _NONMAXSUPPRESSIONCALCULATOROPTIONS_OVERLAPTYPE._serialized_start=642
  _NONMAXSUPPRESSIONCALCULATOROPTIONS_OVERLAPTYPE._serialized_end=749
  _NONMAXSUPPRESSIONCALCULATOROPTIONS_NMSALGORITHM._serialized_start=751
  _NONMAXSUPPRESSIONCALCULATOROPTIONS_NMSALGORITHM._serialized_end=810
# @@protoc_insertion_point(module_scope)