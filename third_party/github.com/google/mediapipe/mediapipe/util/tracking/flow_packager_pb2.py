# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/util/tracking/flow_packager.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.util.tracking import motion_models_pb2 as mediapipe_dot_util_dot_tracking_dot_motion__models__pb2
from mediapipe.util.tracking import region_flow_pb2 as mediapipe_dot_util_dot_tracking_dot_region__flow__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n+mediapipe/util/tracking/flow_packager.proto\x12\tmediapipe\x1a+mediapipe/util/tracking/motion_models.proto\x1a)mediapipe/util/tracking/region_flow.proto\"\xce\x07\n\x0cTrackingData\x12\"\n\x0b\x66rame_flags\x18\x01 \x01(\x05:\x01\x30R\nframeFlags\x12!\n\x0c\x64omain_width\x18\x02 \x01(\x05R\x0b\x64omainWidth\x12#\n\rdomain_height\x18\x03 \x01(\x05R\x0c\x64omainHeight\x12$\n\x0c\x66rame_aspect\x18\x06 \x01(\x02:\x01\x31R\x0b\x66rameAspect\x12@\n\x10\x62\x61\x63kground_model\x18\x04 \x01(\x0b\x32\x15.mediapipe.HomographyR\x0f\x62\x61\x63kgroundModel\x12\x43\n\x0bmotion_data\x18\x05 \x01(\x0b\x32\".mediapipe.TrackingData.MotionDataR\nmotionData\x12\x30\n\x14global_feature_count\x18\x07 \x01(\rR\x12globalFeatureCount\x12\x38\n\x18\x61verage_motion_magnitude\x18\x08 \x01(\x02R\x16\x61verageMotionMagnitude\x1a\xd5\x02\n\nMotionData\x12!\n\x0cnum_elements\x18\x01 \x01(\x05R\x0bnumElements\x12#\n\x0bvector_data\x18\x02 \x03(\x02\x42\x02\x10\x01R\nvectorData\x12\x1d\n\x08track_id\x18\x03 \x03(\x05\x42\x02\x10\x01R\x07trackId\x12#\n\x0brow_indices\x18\x04 \x03(\x05\x42\x02\x10\x01R\nrowIndices\x12!\n\ncol_starts\x18\x05 \x03(\x05\x42\x02\x10\x01R\tcolStarts\x12S\n\x13\x66\x65\x61ture_descriptors\x18\x06 \x03(\x0b\x32\".mediapipe.BinaryFeatureDescriptorR\x12\x66\x65\x61tureDescriptors\x12\x43\n\x1e\x61\x63tively_discarded_tracked_ids\x18\x07 \x03(\x05R\x1b\x61\x63tivelyDiscardedTrackedIds\"\xe0\x01\n\nFrameFlags\x12\"\n\x1eTRACKING_FLAG_PROFILE_BASELINE\x10\x00\x12\x1e\n\x1aTRACKING_FLAG_PROFILE_HIGH\x10\x01\x12\'\n#TRACKING_FLAG_HIGH_FIDELITY_VECTORS\x10\x02\x12%\n!TRACKING_FLAG_BACKGROUND_UNSTABLE\x10\x04\x12\x1c\n\x18TRACKING_FLAG_DUPLICATED\x10\x08\x12 \n\x1cTRACKING_FLAG_CHUNK_BOUNDARY\x10\x10\"\xd3\x02\n\x11TrackingDataChunk\x12\x35\n\x04item\x18\x01 \x03(\x0b\x32!.mediapipe.TrackingDataChunk.ItemR\x04item\x12$\n\nlast_chunk\x18\x02 \x01(\x08:\x05\x66\x61lseR\tlastChunk\x12&\n\x0b\x66irst_chunk\x18\x03 \x01(\x08:\x05\x66\x61lseR\nfirstChunk\x1a\xb8\x01\n\x04Item\x12<\n\rtracking_data\x18\x01 \x01(\x0b\x32\x17.mediapipe.TrackingDataR\x0ctrackingData\x12\x1b\n\tframe_idx\x18\x02 \x01(\x05R\x08\x66rameIdx\x12%\n\x0etimestamp_usec\x18\x03 \x01(\x03R\rtimestampUsec\x12.\n\x13prev_timestamp_usec\x18\x04 \x01(\x03R\x11prevTimestampUsec\"(\n\x12\x42inaryTrackingData\x12\x12\n\x04\x64\x61ta\x18\x01 \x01(\x0cR\x04\x64\x61ta\"\xb7\x01\n\x08MetaData\x12\x1d\n\nnum_frames\x18\x02 \x01(\x07R\tnumFrames\x12\x44\n\rtrack_offsets\x18\x03 \x03(\x0b\x32\x1f.mediapipe.MetaData.TrackOffsetR\x0ctrackOffsets\x1a\x46\n\x0bTrackOffset\x12\x12\n\x04msec\x18\x01 \x01(\x07R\x04msec\x12#\n\rstream_offset\x18\x02 \x01(\x07R\x0cstreamOffset\"p\n\x11TrackingContainer\x12\x16\n\x06header\x18\x01 \x01(\tR\x06header\x12\x1b\n\x07version\x18\x02 \x01(\x07:\x01\x31R\x07version\x12\x12\n\x04size\x18\x03 \x01(\x07R\x04size\x12\x12\n\x04\x64\x61ta\x18\x04 \x01(\x0cR\x04\x64\x61ta\"\xcc\x01\n\x17TrackingContainerFormat\x12\x39\n\tmeta_data\x18\x01 \x01(\x0b\x32\x1c.mediapipe.TrackingContainerR\x08metaData\x12;\n\ntrack_data\x18\x02 \x03(\x0b\x32\x1c.mediapipe.TrackingContainerR\ttrackData\x12\x39\n\tterm_data\x18\x03 \x01(\x0b\x32\x1c.mediapipe.TrackingContainerR\x08termData\"\x88\x01\n\x16TrackingContainerProto\x12\x30\n\tmeta_data\x18\x01 \x01(\x0b\x32\x13.mediapipe.MetaDataR\x08metaData\x12<\n\ntrack_data\x18\x02 \x03(\x0b\x32\x1d.mediapipe.BinaryTrackingDataR\ttrackData\"\xbb\x03\n\x13\x46lowPackagerOptions\x12&\n\x0c\x64omain_width\x18\x01 \x01(\x05:\x03\x32\x35\x36R\x0b\x64omainWidth\x12(\n\rdomain_height\x18\x02 \x01(\x05:\x03\x31\x39\x32R\x0c\x64omainHeight\x12\x45\n\x1c\x62inary_tracking_data_support\x18\x06 \x01(\x08:\x04trueR\x19\x62inaryTrackingDataSupport\x12/\n\x10use_high_profile\x18\x03 \x01(\x08:\x05\x66\x61lseR\x0euseHighProfile\x12\x41\n\x1ahigh_fidelity_16bit_encode\x18\x04 \x01(\x08:\x04trueR\x17highFidelity16bitEncode\x12\x44\n\x1chigh_profile_reuse_threshold\x18\x05 \x01(\x02:\x03\x30.5R\x19highProfileReuseThreshold\"Q\n\x13HighProfileEncoding\x12\x11\n\x0c\x41\x44VANCE_FLAG\x10\x80\x01\x12\x17\n\x13\x44OUBLE_INDEX_ENCODE\x10@\x12\x0e\n\nINDEX_MASK\x10?B5Z3github.com/google/mediapipe/mediapipe/util/tracking')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.util.tracking.flow_packager_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z3github.com/google/mediapipe/mediapipe/util/tracking'
  _TRACKINGDATA_MOTIONDATA.fields_by_name['vector_data']._options = None
  _TRACKINGDATA_MOTIONDATA.fields_by_name['vector_data']._serialized_options = b'\020\001'
  _TRACKINGDATA_MOTIONDATA.fields_by_name['track_id']._options = None
  _TRACKINGDATA_MOTIONDATA.fields_by_name['track_id']._serialized_options = b'\020\001'
  _TRACKINGDATA_MOTIONDATA.fields_by_name['row_indices']._options = None
  _TRACKINGDATA_MOTIONDATA.fields_by_name['row_indices']._serialized_options = b'\020\001'
  _TRACKINGDATA_MOTIONDATA.fields_by_name['col_starts']._options = None
  _TRACKINGDATA_MOTIONDATA.fields_by_name['col_starts']._serialized_options = b'\020\001'
  _TRACKINGDATA._serialized_start=147
  _TRACKINGDATA._serialized_end=1121
  _TRACKINGDATA_MOTIONDATA._serialized_start=553
  _TRACKINGDATA_MOTIONDATA._serialized_end=894
  _TRACKINGDATA_FRAMEFLAGS._serialized_start=897
  _TRACKINGDATA_FRAMEFLAGS._serialized_end=1121
  _TRACKINGDATACHUNK._serialized_start=1124
  _TRACKINGDATACHUNK._serialized_end=1463
  _TRACKINGDATACHUNK_ITEM._serialized_start=1279
  _TRACKINGDATACHUNK_ITEM._serialized_end=1463
  _BINARYTRACKINGDATA._serialized_start=1465
  _BINARYTRACKINGDATA._serialized_end=1505
  _METADATA._serialized_start=1508
  _METADATA._serialized_end=1691
  _METADATA_TRACKOFFSET._serialized_start=1621
  _METADATA_TRACKOFFSET._serialized_end=1691
  _TRACKINGCONTAINER._serialized_start=1693
  _TRACKINGCONTAINER._serialized_end=1805
  _TRACKINGCONTAINERFORMAT._serialized_start=1808
  _TRACKINGCONTAINERFORMAT._serialized_end=2012
  _TRACKINGCONTAINERPROTO._serialized_start=2015
  _TRACKINGCONTAINERPROTO._serialized_end=2151
  _FLOWPACKAGEROPTIONS._serialized_start=2154
  _FLOWPACKAGEROPTIONS._serialized_end=2597
  _FLOWPACKAGEROPTIONS_HIGHPROFILEENCODING._serialized_start=2516
  _FLOWPACKAGEROPTIONS_HIGHPROFILEENCODING._serialized_end=2597
# @@protoc_insertion_point(module_scope)
