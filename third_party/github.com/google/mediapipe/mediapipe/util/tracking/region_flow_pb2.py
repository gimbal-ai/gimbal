# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/util/tracking/region_flow.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n)mediapipe/util/tracking/region_flow.proto\x12\tmediapipe\"%\n\x0fPatchDescriptor\x12\x12\n\x04\x64\x61ta\x18\x01 \x03(\x02R\x04\x64\x61ta\"-\n\x17\x42inaryFeatureDescriptor\x12\x12\n\x04\x64\x61ta\x18\x01 \x01(\x0cR\x04\x64\x61ta\"Y\n\x15TemporalIRLSSmoothing\x12 \n\nweight_sum\x18\x01 \x01(\x02:\x01\x30R\tweightSum\x12\x1e\n\tvalue_sum\x18\x02 \x01(\x02:\x01\x30R\x08valueSum\"\xf8\x05\n\x11RegionFlowFeature\x12\x0f\n\x01x\x18\x01 \x01(\x02:\x01\x30R\x01x\x12\x0f\n\x01y\x18\x02 \x01(\x02:\x01\x30R\x01y\x12\x11\n\x02\x64x\x18\x03 \x01(\x02:\x01\x30R\x02\x64x\x12\x11\n\x02\x64y\x18\x04 \x01(\x02:\x01\x30R\x02\x64y\x12\x1d\n\x08track_id\x18\r \x01(\x05:\x02-1R\x07trackId\x12(\n\x0etracking_error\x18\x05 \x01(\x02:\x01\x30R\rtrackingError\x12\"\n\x0birls_weight\x18\x06 \x01(\x02:\x01\x31R\nirlsWeight\x12*\n\x0f\x63orner_response\x18\x0b \x01(\x02:\x01\x30R\x0e\x63ornerResponse\x12I\n\x12\x66\x65\x61ture_descriptor\x18\x07 \x01(\x0b\x32\x1a.mediapipe.PatchDescriptorR\x11\x66\x65\x61tureDescriptor\x12T\n\x18\x66\x65\x61ture_match_descriptor\x18\x08 \x01(\x0b\x32\x1a.mediapipe.PatchDescriptorR\x16\x66\x65\x61tureMatchDescriptor\x12\x45\n\rinternal_irls\x18\n \x01(\x0b\x32 .mediapipe.TemporalIRLSSmoothingR\x0cinternalIrls\x12\x14\n\x05label\x18\x0e \x01(\tR\x05label\x12\x14\n\x05\x66lags\x18\x0f \x01(\x05R\x05\x66lags\x12\x1d\n\nfeature_id\x18\x10 \x01(\x05R\tfeatureId\x12\x19\n\x06octave\x18\x11 \x01(\x05:\x01\x30R\x06octave\x12^\n\x19\x62inary_feature_descriptor\x18\x12 \x01(\x0b\x32\".mediapipe.BinaryFeatureDescriptorR\x17\x62inaryFeatureDescriptor\"H\n\x05\x46lags\x12\x1c\n\x18REGION_FLOW_FLAG_UNKNOWN\x10\x00\x12!\n\x1dREGION_FLOW_FLAG_BROKEN_TRACK\x10\x01*\x04\x08\t\x10\n*\x04\x08\x0c\x10\r\"\x88\x06\n\x0fRegionFlowFrame\x12\x46\n\x0bregion_flow\x18\x01 \x03(\x0b\x32%.mediapipe.RegionFlowFrame.RegionFlowR\nregionFlow\x12/\n\x12num_total_features\x18\x02 \x01(\x05:\x01\x30R\x10numTotalFeatures\x12,\n\x0eunstable_frame\x18\x04 \x01(\x08:\x05\x66\x61lseR\runstableFrame\x12\x1d\n\nblur_score\x18\x07 \x01(\x02R\tblurScore\x12\x1f\n\x0b\x66rame_width\x18\x08 \x01(\x05R\nframeWidth\x12!\n\x0c\x66rame_height\x18\t \x01(\x05R\x0b\x66rameHeight\x12U\n\x10\x62lock_descriptor\x18\n \x01(\x0b\x32*.mediapipe.RegionFlowFrame.BlockDescriptorR\x0f\x62lockDescriptor\x1a\xdf\x01\n\nRegionFlow\x12\x1b\n\tregion_id\x18\x01 \x02(\x05R\x08regionId\x12 \n\ncentroid_x\x18\x02 \x01(\x02:\x01\x30R\tcentroidX\x12 \n\ncentroid_y\x18\x03 \x01(\x02:\x01\x30R\tcentroidY\x12\x18\n\x06\x66low_x\x18\x04 \x01(\x02:\x01\x30R\x05\x66lowX\x12\x18\n\x06\x66low_y\x18\x05 \x01(\x02:\x01\x30R\x05\x66lowY\x12\x36\n\x07\x66\x65\x61ture\x18\x07 \x03(\x0b\x32\x1c.mediapipe.RegionFlowFeatureR\x07\x66\x65\x61ture*\x04\x08\x06\x10\x07\x1a\x9f\x01\n\x0f\x42lockDescriptor\x12\x1f\n\x0b\x62lock_width\x18\x01 \x01(\x05R\nblockWidth\x12!\n\x0c\x62lock_height\x18\x02 \x01(\x05R\x0b\x62lockHeight\x12#\n\x0cnum_blocks_x\x18\x03 \x01(\x05:\x01\x30R\nnumBlocksX\x12#\n\x0cnum_blocks_y\x18\x04 \x01(\x05:\x01\x30R\nnumBlocksY*\x04\x08\x03\x10\x04*\x04\x08\x05\x10\x06*\x04\x08\x06\x10\x07\"\xe6\x04\n\x15RegionFlowFeatureList\x12\x36\n\x07\x66\x65\x61ture\x18\x01 \x03(\x0b\x32\x1c.mediapipe.RegionFlowFeatureR\x07\x66\x65\x61ture\x12\x1f\n\x0b\x66rame_width\x18\x02 \x01(\x05R\nframeWidth\x12!\n\x0c\x66rame_height\x18\x03 \x01(\x05R\x0b\x66rameHeight\x12!\n\x08unstable\x18\x04 \x01(\x08:\x05\x66\x61lseR\x08unstable\x12\x33\n\x14\x64istance_from_border\x18\x05 \x01(\x05:\x01\x30R\x12\x64istanceFromBorder\x12\x1d\n\nblur_score\x18\x06 \x01(\x02R\tblurScore\x12&\n\x0blong_tracks\x18\x07 \x01(\x08:\x05\x66\x61lseR\nlongTracks\x12@\n\x1b\x66rac_long_features_rejected\x18\x08 \x01(\x02:\x01\x30R\x18\x66racLongFeaturesRejected\x12\x31\n\x12visual_consistency\x18\t \x01(\x02:\x02-1R\x11visualConsistency\x12(\n\x0etimestamp_usec\x18\n \x01(\x03:\x01\x30R\rtimestampUsec\x12\"\n\x0bmatch_frame\x18\x0b \x01(\x05:\x01\x30R\nmatchFrame\x12*\n\ris_duplicated\x18\x0c \x01(\x08:\x05\x66\x61lseR\x0cisDuplicated\x12\x43\n\x1e\x61\x63tively_discarded_tracked_ids\x18\r \x03(\x05R\x1b\x61\x63tivelyDiscardedTrackedIds\"\xb6\x04\n\x0cSalientPoint\x12#\n\x0cnorm_point_x\x18\x01 \x01(\x02:\x01\x30R\nnormPointX\x12#\n\x0cnorm_point_y\x18\x02 \x01(\x02:\x01\x30R\nnormPointY\x12X\n\x04type\x18\x0b \x01(\x0e\x32(.mediapipe.SalientPoint.SalientPointType:\x1aSALIENT_POINT_TYPE_INCLUDER\x04type\x12\x17\n\x04left\x18\x03 \x01(\x02:\x03\x30.3R\x04left\x12\x1b\n\x06\x62ottom\x18\x04 \x01(\x02:\x03\x30.3R\x06\x62ottom\x12\x19\n\x05right\x18\t \x01(\x02:\x03\x30.3R\x05right\x12\x15\n\x03top\x18\n \x01(\x02:\x03\x30.3R\x03top\x12\x1a\n\x06weight\x18\x05 \x01(\x02:\x02\x31\x35R\x06weight\x12\x1d\n\nnorm_major\x18\x06 \x01(\x02R\tnormMajor\x12\x1d\n\nnorm_minor\x18\x07 \x01(\x02R\tnormMinor\x12\x14\n\x05\x61ngle\x18\x08 \x01(\x02R\x05\x61ngle\"\x9d\x01\n\x10SalientPointType\x12\x1e\n\x1aSALIENT_POINT_TYPE_UNKNOWN\x10\x00\x12\x1e\n\x1aSALIENT_POINT_TYPE_INCLUDE\x10\x01\x12#\n\x1fSALIENT_POINT_TYPE_EXCLUDE_LEFT\x10\x02\x12$\n SALIENT_POINT_TYPE_EXCLUDE_RIGHT\x10\x03*\n\x08\xa0\x9c\x01\x10\x80\x80\x80\x80\x02\"N\n\x11SalientPointFrame\x12-\n\x05point\x18\x01 \x03(\x0b\x32\x17.mediapipe.SalientPointR\x05point*\n\x08\xa0\x9c\x01\x10\x80\x80\x80\x80\x02\x42V\n\x1d\x63om.google.mediapipe.trackingP\x01Z3github.com/google/mediapipe/mediapipe/util/tracking')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.util.tracking.region_flow_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'\n\035com.google.mediapipe.trackingP\001Z3github.com/google/mediapipe/mediapipe/util/tracking'
  _PATCHDESCRIPTOR._serialized_start=56
  _PATCHDESCRIPTOR._serialized_end=93
  _BINARYFEATUREDESCRIPTOR._serialized_start=95
  _BINARYFEATUREDESCRIPTOR._serialized_end=140
  _TEMPORALIRLSSMOOTHING._serialized_start=142
  _TEMPORALIRLSSMOOTHING._serialized_end=231
  _REGIONFLOWFEATURE._serialized_start=234
  _REGIONFLOWFEATURE._serialized_end=994
  _REGIONFLOWFEATURE_FLAGS._serialized_start=910
  _REGIONFLOWFEATURE_FLAGS._serialized_end=982
  _REGIONFLOWFRAME._serialized_start=997
  _REGIONFLOWFRAME._serialized_end=1773
  _REGIONFLOWFRAME_REGIONFLOW._serialized_start=1370
  _REGIONFLOWFRAME_REGIONFLOW._serialized_end=1593
  _REGIONFLOWFRAME_BLOCKDESCRIPTOR._serialized_start=1596
  _REGIONFLOWFRAME_BLOCKDESCRIPTOR._serialized_end=1755
  _REGIONFLOWFEATURELIST._serialized_start=1776
  _REGIONFLOWFEATURELIST._serialized_end=2390
  _SALIENTPOINT._serialized_start=2393
  _SALIENTPOINT._serialized_end=2959
  _SALIENTPOINT_SALIENTPOINTTYPE._serialized_start=2790
  _SALIENTPOINT_SALIENTPOINTTYPE._serialized_end=2947
  _SALIENTPOINTFRAME._serialized_start=2961
  _SALIENTPOINTFRAME._serialized_end=3039
# @@protoc_insertion_point(module_scope)
