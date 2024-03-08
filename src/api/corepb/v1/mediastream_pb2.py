# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/api/corepb/v1/mediastream.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n#src/api/corepb/v1/mediastream.proto\x12\x18gml.internal.api.core.v1\x1a\x14gogoproto/gogo.proto\"3\n\x05Label\x12\x14\n\x05label\x18\x01 \x01(\tR\x05label\x12\x14\n\x05score\x18\x02 \x01(\x02R\x05score\"d\n\x14NormalizedCenterRect\x12\x0e\n\x02xc\x18\x01 \x01(\x02R\x02xc\x12\x0e\n\x02yc\x18\x02 \x01(\x02R\x02yc\x12\x14\n\x05width\x18\x03 \x01(\x02R\x05width\x12\x16\n\x06height\x18\x04 \x01(\x02R\x06height\"\x95\x01\n\tDetection\x12\x35\n\x05label\x18\x01 \x03(\x0b\x32\x1f.gml.internal.api.core.v1.LabelR\x05label\x12Q\n\x0c\x62ounding_box\x18\x02 \x01(\x0b\x32..gml.internal.api.core.v1.NormalizedCenterRectR\x0b\x62oundingBox\"R\n\rDetectionList\x12\x41\n\tdetection\x18\x01 \x03(\x0b\x32#.gml.internal.api.core.v1.DetectionR\tdetection\"\xbb\x01\n\x0eImageHistogram\x12\x45\n\x07\x63hannel\x18\x01 \x01(\x0e\x32+.gml.internal.api.core.v1.ImageColorChannelR\x07\x63hannel\x12\x10\n\x03min\x18\x02 \x01(\x01R\x03min\x12\x10\n\x03max\x18\x03 \x01(\x01R\x03max\x12\x10\n\x03num\x18\x04 \x01(\x03R\x03num\x12\x10\n\x03sum\x18\x05 \x01(\x01R\x03sum\x12\x1a\n\x06\x62ucket\x18\x06 \x03(\x03\x42\x02\x10\x01R\x06\x62ucket\"_\n\x13ImageHistogramBatch\x12H\n\nhistograms\x18\x01 \x03(\x0b\x32(.gml.internal.api.core.v1.ImageHistogramR\nhistograms\"e\n\x13ImageQualityMetrics\x12#\n\rbrisque_score\x18\x01 \x01(\x01R\x0c\x62risqueScore\x12)\n\x10\x62lurriness_score\x18\x02 \x01(\x01R\x0f\x62lurrinessScore\"\xe0\x02\n\x11ImageOverlayChunk\x12\x31\n\x08\x66rame_ts\x18\x01 \x01(\x03\x42\x16\xe2\xde\x1f\x07\x46rameTS\xea\xde\x1f\x07\x66rameTSR\x07\x66rameTS\x12\x19\n\x03\x65of\x18\x02 \x01(\x08\x42\x07\xe2\xde\x1f\x03\x45OFR\x03\x65of\x12I\n\ndetections\x18\x64 \x01(\x0b\x32\'.gml.internal.api.core.v1.DetectionListH\x00R\ndetections\x12P\n\nhistograms\x18\xc8\x01 \x01(\x0b\x32-.gml.internal.api.core.v1.ImageHistogramBatchH\x00R\nhistograms\x12U\n\rimage_quality\x18\xac\x02 \x01(\x0b\x32-.gml.internal.api.core.v1.ImageQualityMetricsH\x00R\x0cimageQualityB\t\n\x07overlay\"\x81\x01\n\tH264Chunk\x12\x31\n\x08\x66rame_ts\x18\x01 \x01(\x03\x42\x16\xe2\xde\x1f\x07\x46rameTS\xea\xde\x1f\x07\x66rameTSR\x07\x66rameTS\x12\x19\n\x03\x65of\x18\x02 \x01(\x08\x42\x07\xe2\xde\x1f\x03\x45OFR\x03\x65of\x12&\n\x08nal_data\x18\x03 \x01(\x0c\x42\x0b\xe2\xde\x1f\x07NALDataR\x07nalData\"Z\n\x0bVideoHeader\x12\x14\n\x05width\x18\x01 \x01(\x03R\x05width\x12\x16\n\x06height\x18\x02 \x01(\x03R\x06height\x12\x1d\n\nframe_rate\x18\x03 \x01(\x01R\tframeRate*\xac\x01\n\x11ImageColorChannel\x12\x1f\n\x1bIMAGE_COLOR_CHANNEL_UNKNOWN\x10\x00\x12\x1c\n\x18IMAGE_COLOR_CHANNEL_GRAY\x10\x01\x12\x1b\n\x17IMAGE_COLOR_CHANNEL_RED\x10\x02\x12\x1d\n\x19IMAGE_COLOR_CHANNEL_GREEN\x10\x03\x12\x1c\n\x18IMAGE_COLOR_CHANNEL_BLUE\x10\x04\x42/Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.api.corepb.v1.mediastream_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z-gimletlabs.ai/gimlet/src/api/corepb/v1;corepb'
  _IMAGEHISTOGRAM.fields_by_name['bucket']._options = None
  _IMAGEHISTOGRAM.fields_by_name['bucket']._serialized_options = b'\020\001'
  _IMAGEOVERLAYCHUNK.fields_by_name['frame_ts']._options = None
  _IMAGEOVERLAYCHUNK.fields_by_name['frame_ts']._serialized_options = b'\342\336\037\007FrameTS\352\336\037\007frameTS'
  _IMAGEOVERLAYCHUNK.fields_by_name['eof']._options = None
  _IMAGEOVERLAYCHUNK.fields_by_name['eof']._serialized_options = b'\342\336\037\003EOF'
  _H264CHUNK.fields_by_name['frame_ts']._options = None
  _H264CHUNK.fields_by_name['frame_ts']._serialized_options = b'\342\336\037\007FrameTS\352\336\037\007frameTS'
  _H264CHUNK.fields_by_name['eof']._options = None
  _H264CHUNK.fields_by_name['eof']._serialized_options = b'\342\336\037\003EOF'
  _H264CHUNK.fields_by_name['nal_data']._options = None
  _H264CHUNK.fields_by_name['nal_data']._serialized_options = b'\342\336\037\007NALData'
  _IMAGECOLORCHANNEL._serialized_start=1448
  _IMAGECOLORCHANNEL._serialized_end=1620
  _LABEL._serialized_start=87
  _LABEL._serialized_end=138
  _NORMALIZEDCENTERRECT._serialized_start=140
  _NORMALIZEDCENTERRECT._serialized_end=240
  _DETECTION._serialized_start=243
  _DETECTION._serialized_end=392
  _DETECTIONLIST._serialized_start=394
  _DETECTIONLIST._serialized_end=476
  _IMAGEHISTOGRAM._serialized_start=479
  _IMAGEHISTOGRAM._serialized_end=666
  _IMAGEHISTOGRAMBATCH._serialized_start=668
  _IMAGEHISTOGRAMBATCH._serialized_end=763
  _IMAGEQUALITYMETRICS._serialized_start=765
  _IMAGEQUALITYMETRICS._serialized_end=866
  _IMAGEOVERLAYCHUNK._serialized_start=869
  _IMAGEOVERLAYCHUNK._serialized_end=1221
  _H264CHUNK._serialized_start=1224
  _H264CHUNK._serialized_end=1353
  _VIDEOHEADER._serialized_start=1355
  _VIDEOHEADER._serialized_end=1445
# @@protoc_insertion_point(module_scope)