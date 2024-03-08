# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/calculators/image/scale_image_calculator.proto
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
from mediapipe.framework.formats import image_format_pb2 as mediapipe_dot_framework_dot_formats_dot_image__format__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n8mediapipe/calculators/image/scale_image_calculator.proto\x12\tmediapipe\x1a$mediapipe/framework/calculator.proto\x1a.mediapipe/framework/formats/image_format.proto\"\xfb\x08\n\x1bScaleImageCalculatorOptions\x12!\n\x0ctarget_width\x18\x01 \x01(\x05R\x0btargetWidth\x12#\n\rtarget_height\x18\x02 \x01(\x05R\x0ctargetHeight\x12&\n\x0ftarget_max_area\x18\x0f \x01(\x05R\rtargetMaxArea\x12\x38\n\x15preserve_aspect_ratio\x18\x03 \x01(\x08:\x04trueR\x13preserveAspectRatio\x12.\n\x10min_aspect_ratio\x18\x04 \x01(\t:\x04\x39/16R\x0eminAspectRatio\x12.\n\x10max_aspect_ratio\x18\x05 \x01(\t:\x04\x31\x36/9R\x0emaxAspectRatio\x12\x42\n\routput_format\x18\x06 \x01(\x0e\x32\x1d.mediapipe.ImageFormat.FormatR\x0coutputFormat\x12g\n\talgorithm\x18\x07 \x01(\x0e\x32\x35.mediapipe.ScaleImageCalculatorOptions.ScaleAlgorithm:\x12SCALE_ALGO_DEFAULTR\talgorithm\x12\x31\n\x12\x61lignment_boundary\x18\x08 \x01(\x05:\x02\x31\x36R\x11\x61lignmentBoundary\x12\x38\n\x15set_alignment_padding\x18\t \x01(\x08:\x04trueR\x13setAlignmentPadding\x12S\n#OBSOLETE_skip_linear_rgb_conversion\x18\n \x01(\x08:\x05\x66\x61lseR\x1fOBSOLETESkipLinearRgbConversion\x12\x41\n\x1bpost_sharpening_coefficient\x18\x0b \x01(\x02:\x01\x30R\x19postSharpeningCoefficient\x12@\n\x0cinput_format\x18\x0c \x01(\x0e\x32\x1d.mediapipe.ImageFormat.FormatR\x0binputFormat\x12\x32\n\x14scale_to_multiple_of\x18\r \x01(\x05:\x01\x32R\x11scaleToMultipleOf\x12\"\n\tuse_bt709\x18\x0e \x01(\x08:\x05\x66\x61lseR\x08useBt709\"\xaa\x01\n\x0eScaleAlgorithm\x12\x16\n\x12SCALE_ALGO_DEFAULT\x10\x00\x12\x15\n\x11SCALE_ALGO_LINEAR\x10\x01\x12\x14\n\x10SCALE_ALGO_CUBIC\x10\x02\x12\x13\n\x0fSCALE_ALGO_AREA\x10\x03\x12\x16\n\x12SCALE_ALGO_LANCZOS\x10\x04\x12&\n\"SCALE_ALGO_DEFAULT_WITHOUT_UPSCALE\x10\x05\x32Y\n\x03\x65xt\x12\x1c.mediapipe.CalculatorOptions\x18\xbb\xe5\xca\x1f \x01(\x0b\x32&.mediapipe.ScaleImageCalculatorOptionsR\x03\x65xtB9Z7github.com/google/mediapipe/mediapipe/calculators/image')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.calculators.image.scale_image_calculator_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_calculator__options__pb2.CalculatorOptions.RegisterExtension(_SCALEIMAGECALCULATOROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z7github.com/google/mediapipe/mediapipe/calculators/image'
  _SCALEIMAGECALCULATOROPTIONS._serialized_start=158
  _SCALEIMAGECALCULATOROPTIONS._serialized_end=1305
  _SCALEIMAGECALCULATOROPTIONS_SCALEALGORITHM._serialized_start=1044
  _SCALEIMAGECALCULATOROPTIONS_SCALEALGORITHM._serialized_end=1214
# @@protoc_insertion_point(module_scope)