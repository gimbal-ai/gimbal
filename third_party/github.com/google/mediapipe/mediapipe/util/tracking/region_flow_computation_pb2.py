# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/util/tracking/region_flow_computation.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.util.tracking import tone_estimation_pb2 as mediapipe_dot_util_dot_tracking_dot_tone__estimation__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n5mediapipe/util/tracking/region_flow_computation.proto\x12\tmediapipe\x1a-mediapipe/util/tracking/tone_estimation.proto\"\xa0\x18\n\x0fTrackingOptions\x12\x81\x01\n\x1binternal_tracking_direction\x18\x13 \x01(\x0e\x32(.mediapipe.TrackingOptions.FlowDirection:\x17\x46LOW_DIRECTION_BACKWARDR\x19internalTrackingDirection\x12u\n\x15output_flow_direction\x18\x14 \x01(\x0e\x32(.mediapipe.TrackingOptions.FlowDirection:\x17\x46LOW_DIRECTION_BACKWARDR\x13outputFlowDirection\x12p\n\x0ftracking_policy\x18\x19 \x01(\x0e\x32).mediapipe.TrackingOptions.TrackingPolicy:\x1cTRACKING_POLICY_SINGLE_FRAMER\x0etrackingPolicy\x12\x34\n\x15multi_frames_to_track\x18\x01 \x01(\x05:\x01\x31R\x12multiFramesToTrack\x12\x38\n\x16long_tracks_max_frames\x18\x1a \x01(\x05:\x03\x33\x30\x30R\x13longTracksMaxFrames\x12\'\n\x0cmax_features\x18\x02 \x01(\x05:\x04\x32\x30\x30\x30R\x0bmaxFeatures\x12\x91\x01\n\x18\x63orner_extraction_method\x18\x1b \x01(\x0e\x32\x31.mediapipe.TrackingOptions.CornerExtractionMethod:$CORNER_EXTRACTION_METHOD_MIN_EIG_VALR\x16\x63ornerExtractionMethod\x12g\n\x14min_eig_val_settings\x18\x1c \x01(\x0b\x32\x36.mediapipe.TrackingOptions.MinEigValExtractionSettingsR\x11minEigValSettings\x12\\\n\x0fharris_settings\x18\x1d \x01(\x0b\x32\x33.mediapipe.TrackingOptions.HarrisExtractionSettingsR\x0eharrisSettings\x12V\n\rfast_settings\x18\x1f \x01(\x0b\x32\x31.mediapipe.TrackingOptions.FastExtractionSettingsR\x0c\x66\x61stSettings\x12\x34\n\x14tracking_window_size\x18\x04 \x01(\x05:\x02\x31\x30R\x12trackingWindowSize\x12\x33\n\x13tracking_iterations\x18\x05 \x01(\x05:\x02\x31\x30R\x12trackingIterations\x12\x46\n\x1c\x66ractional_tracking_distance\x18\x06 \x01(\x02:\x04\x30.15R\x1a\x66ractionalTrackingDistance\x12\x43\n\x1a\x61\x64\x61ptive_tracking_distance\x18\x18 \x01(\x08:\x05\x66\x61lseR\x18\x61\x64\x61ptiveTrackingDistance\x12\x33\n\x14min_feature_distance\x18\x07 \x01(\x02:\x01\x37R\x12minFeatureDistance\x12<\n\x17\x64istance_downscale_sqrt\x18\x15 \x01(\x08:\x04trueR\x15\x64istanceDownscaleSqrt\x12J\n\x1f\x61\x64\x61ptive_good_features_to_track\x18\x08 \x01(\x08:\x04trueR\x1b\x61\x64\x61ptiveGoodFeaturesToTrack\x12\x45\n\x1c\x61\x64\x61ptive_features_block_size\x18\t \x01(\x02:\x04\x30.26R\x19\x61\x64\x61ptiveFeaturesBlockSize\x12;\n\x18\x61\x64\x61ptive_features_levels\x18\n \x01(\x05:\x01\x31R\x16\x61\x64\x61ptiveFeaturesLevels\x12?\n\x1a\x61\x64\x61ptive_extraction_levels\x18\x16 \x01(\x05:\x01\x31R\x18\x61\x64\x61ptiveExtractionLevels\x12U\n&adaptive_extraction_levels_lowest_size\x18\x17 \x01(\x05:\x01\x30R\"adaptiveExtractionLevelsLowestSize\x12J\n\x1fsynthetic_zero_motion_grid_step\x18\r \x01(\x02:\x04\x30.04R\x1bsyntheticZeroMotionGridStep\x12;\n\x16wide_baseline_matching\x18\x0e \x01(\x08:\x05\x66\x61lseR\x14wideBaselineMatching\x12\x35\n\x14ratio_test_threshold\x18\x0f \x01(\x02:\x03\x30.8R\x12ratioTestThreshold\x12\x46\n\x1crefine_wide_baseline_matches\x18\x10 \x01(\x08:\x05\x66\x61lseR\x19refineWideBaselineMatches\x12K\n!reuse_features_max_frame_distance\x18\x11 \x01(\x05:\x01\x30R\x1dreuseFeaturesMaxFrameDistance\x12K\n reuse_features_min_survived_frac\x18\x12 \x01(\x02:\x03\x30.7R\x1creuseFeaturesMinSurvivedFrac\x12}\n\x1aklt_tracker_implementation\x18  \x01(\x0e\x32\x33.mediapipe.TrackingOptions.KltTrackerImplementation:\nKLT_OPENCVR\x18kltTrackerImplementation\x1a\xa1\x01\n\x1bMinEigValExtractionSettings\x12\x38\n\x15\x66\x65\x61ture_quality_level\x18\x01 \x01(\x02:\x04\x30.01R\x13\x66\x65\x61tureQualityLevel\x12H\n\x1d\x61\x64\x61ptive_lowest_quality_level\x18\x02 \x01(\x02:\x05\x38\x65-05R\x1a\x61\x64\x61ptiveLowestQualityLevel\x1aW\n\x18HarrisExtractionSettings\x12;\n\x15\x66\x65\x61ture_quality_level\x18\x01 \x01(\x02:\x07\x30.00025R\x13\x66\x65\x61tureQualityLevel\x1a:\n\x16\x46\x61stExtractionSettings\x12 \n\tthreshold\x18\x01 \x01(\x05:\x02\x31\x30R\tthreshold\"\x86\x01\n\rFlowDirection\x12\x1a\n\x16\x46LOW_DIRECTION_UNKNOWN\x10\x00\x12\x1a\n\x16\x46LOW_DIRECTION_FORWARD\x10\x01\x12\x1b\n\x17\x46LOW_DIRECTION_BACKWARD\x10\x02\x12 \n\x1c\x46LOW_DIRECTION_CONSECUTIVELY\x10\x03\"\x91\x01\n\x0eTrackingPolicy\x12\x1b\n\x17TRACKING_POLICY_UNKNOWN\x10\x00\x12 \n\x1cTRACKING_POLICY_SINGLE_FRAME\x10\x01\x12\x1f\n\x1bTRACKING_POLICY_MULTI_FRAME\x10\x02\x12\x1f\n\x1bTRACKING_POLICY_LONG_TRACKS\x10\x03\"\xb0\x01\n\x16\x43ornerExtractionMethod\x12$\n CORNER_EXTRACTION_METHOD_UNKNOWN\x10\x00\x12#\n\x1f\x43ORNER_EXTRACTION_METHOD_HARRIS\x10\x01\x12(\n$CORNER_EXTRACTION_METHOD_MIN_EIG_VAL\x10\x02\x12!\n\x1d\x43ORNER_EXTRACTION_METHOD_FAST\x10\x03\"?\n\x18KltTrackerImplementation\x12\x13\n\x0fKLT_UNSPECIFIED\x10\x00\x12\x0e\n\nKLT_OPENCV\x10\x01*\x04\x08\x03\x10\x04*\x04\x08\x0b\x10\x0c*\x04\x08\x0c\x10\r*\x04\x08\x1e\x10\x1f\"\x88*\n\x1cRegionFlowComputationOptions\x12\x45\n\x10tracking_options\x18\x01 \x01(\x0b\x32\x1a.mediapipe.TrackingOptionsR\x0ftrackingOptions\x12\x31\n\x13min_feature_inliers\x18\x02 \x01(\x05:\x01\x33R\x11minFeatureInliers\x12\x44\n\x1crelative_min_feature_inliers\x18. \x01(\x02:\x03\x30.2R\x19relativeMinFeatureInliers\x12)\n\x0epre_blur_sigma\x18! \x01(\x02:\x03\x30.8R\x0cpreBlurSigma\x12;\n\x18ransac_rounds_per_region\x18\x03 \x01(\x05:\x02\x31\x35R\x15ransacRoundsPerRegion\x12H\n\x1f\x61\x62solute_inlier_error_threshold\x18\x04 \x01(\x02:\x01\x32R\x1c\x61\x62soluteInlierErrorThreshold\x12@\n\x1b\x66rac_inlier_error_threshold\x18\x34 \x01(\x02:\x01\x30R\x18\x66racInlierErrorThreshold\x12J\n\x1frelative_inlier_error_threshold\x18, \x01(\x02:\x03\x30.1R\x1crelativeInlierErrorThreshold\x12)\n\x0ftop_inlier_sets\x18- \x01(\x05:\x01\x32R\rtopInlierSets\x12\x33\n\x12no_estimation_mode\x18( \x01(\x08:\x05\x66\x61lseR\x10noEstimationMode\x12\x41\n\x1a\x66\x61st_estimation_block_size\x18\x06 \x01(\x02:\x04\x30.25R\x17\x66\x61stEstimationBlockSize\x12G\n\x1e\x66\x61st_estimation_min_block_size\x18\x19 \x01(\x05:\x03\x31\x30\x30R\x1a\x66\x61stEstimationMinBlockSize\x12\x44\n\x1d\x66\x61st_estimation_overlap_grids\x18\x16 \x01(\x05:\x01\x33R\x1a\x66\x61stEstimationOverlapGrids\x12\x46\n\x1dmax_magnitude_threshold_ratio\x18\x17 \x01(\x02:\x03\x30.2R\x1amaxMagnitudeThresholdRatio\x12\x39\n\x17median_magnitude_bounds\x18\x33 \x01(\x02:\x01\x30R\x15medianMagnitudeBounds\x12\x82\x01\n\x13irls_initialization\x18\x31 \x01(\x0e\x32:.mediapipe.RegionFlowComputationOptions.IrlsInitialization:\x15IRIS_INIT_CONSISTENCYR\x12irlsInitialization\x12u\n\x0f\x64ownsample_mode\x18\x0b \x01(\x0e\x32\x36.mediapipe.RegionFlowComputationOptions.DownsampleMode:\x14\x44OWNSAMPLE_MODE_NONER\x0e\x64ownsampleMode\x12\x30\n\x11\x64ownsampling_size\x18\x0c \x01(\x05:\x03\x32\x35\x36R\x10\x64ownsamplingSize\x12.\n\x11\x64ownsample_factor\x18\x12 \x01(\x02:\x01\x32R\x10\x64ownsampleFactor\x12=\n\x17round_downsample_factor\x18> \x01(\x08:\x05\x66\x61lseR\x15roundDownsampleFactor\x12k\n\x13\x64ownsample_schedule\x18\x13 \x01(\x0b\x32:.mediapipe.RegionFlowComputationOptions.DownSampleScheduleR\x12\x64ownsampleSchedule\x12:\n\x17min_feature_requirement\x18\r \x01(\x05:\x02\x32\x30R\x15minFeatureRequirement\x12\x30\n\x11min_feature_cover\x18\x0e \x01(\x02:\x04\x30.15R\x0fminFeatureCover\x12\x36\n\x16min_feature_cover_grid\x18\x14 \x01(\x05:\x01\x38R\x13minFeatureCoverGrid\x12\x33\n\x12\x63ompute_blur_score\x18\x11 \x01(\x08:\x05\x66\x61lseR\x10\x63omputeBlurScore\x12\x66\n\x12\x62lur_score_options\x18\x1f \x01(\x0b\x32\x38.mediapipe.RegionFlowComputationOptions.BlurScoreOptionsR\x10\x62lurScoreOptions\x12~\n\x1avisual_consistency_options\x18\x37 \x01(\x0b\x32@.mediapipe.RegionFlowComputationOptions.VisualConsistencyOptionsR\x18visualConsistencyOptions\x12\x39\n\x17patch_descriptor_radius\x18\x15 \x01(\x05:\x01\x33R\x15patchDescriptorRadius\x12\x33\n\x14\x64istance_from_border\x18\x32 \x01(\x05:\x01\x33R\x12\x64istanceFromBorder\x12\x38\n\x15\x63orner_response_scale\x18\x1a \x01(\x02:\x04\x31\x35\x30\x30R\x13\x63ornerResponseScale\x12.\n\x0fverify_features\x18\x1b \x01(\x08:\x05\x66\x61lseR\x0everifyFeatures\x12\x38\n\x15verification_distance\x18\x1c \x01(\x02:\x03\x30.5R\x14verificationDistance\x12\x36\n\x14verify_long_features\x18\x35 \x01(\x08:\x04trueR\x12verifyLongFeatures\x12S\n#long_feature_verification_threshold\x18\x36 \x01(\x02:\x04\x30.04R longFeatureVerificationThreshold\x12\x44\n\x1dmax_long_feature_acceleration\x18\x38 \x01(\x02:\x01\x35R\x1amaxLongFeatureAcceleration\x12N\n verify_long_feature_acceleration\x18? \x01(\x08:\x05\x66\x61lseR\x1dverifyLongFeatureAcceleration\x12K\n!verify_long_feature_trigger_ratio\x18@ \x01(\x02:\x01\x30R\x1dverifyLongFeatureTriggerRatio\x12<\n\x16histogram_equalization\x18\x39 \x01(\x08:\x05\x66\x61lseR\x15histogramEqualization\x12\x61\n+use_synthetic_zero_motion_tracks_all_frames\x18\" \x01(\x08:\x05\x66\x61lseR%useSyntheticZeroMotionTracksAllFrames\x12\x63\n,use_synthetic_zero_motion_tracks_first_frame\x18# \x01(\x08:\x05\x66\x61lseR&useSyntheticZeroMotionTracksFirstFrame\x12.\n\x0fgain_correction\x18$ \x01(\x08:\x05\x66\x61lseR\x0egainCorrection\x12\x37\n\x14\x66\x61st_gain_correction\x18= \x01(\x08:\x05\x66\x61lseR\x12\x66\x61stGainCorrection\x12S\n#gain_correction_multiple_hypotheses\x18/ \x01(\x08:\x04trueR gainCorrectionMultipleHypotheses\x12Y\n\'gain_correction_inlier_improvement_frac\x18\x30 \x01(\x02:\x03\x30.1R#gainCorrectionInlierImprovementFrac\x12N\n gain_correction_bright_reference\x18; \x01(\x08:\x05\x66\x61lseR\x1dgainCorrectionBrightReference\x12J\n gain_correction_triggering_ratio\x18< \x01(\x02:\x01\x30R\x1dgainCorrectionTriggeringRatio\x12\x38\n\x16\x66rac_gain_feature_size\x18% \x01(\x02:\x03\x30.3R\x13\x66racGainFeatureSize\x12)\n\x0e\x66rac_gain_step\x18& \x01(\x02:\x03\x30.1R\x0c\x66racGainStep\x12\x83\x01\n\x11gain_correct_mode\x18) \x01(\x0e\x32\x37.mediapipe.RegionFlowComputationOptions.GainCorrectMode:\x1eGAIN_CORRECT_MODE_DEFAULT_USERR\x0fgainCorrectMode\x12Y\n\x10gain_bias_bounds\x18\' \x01(\x0b\x32/.mediapipe.ToneEstimationOptions.GainBiasBoundsR\x0egainBiasBounds\x12h\n\x0cimage_format\x18: \x01(\x0e\x32\x33.mediapipe.RegionFlowComputationOptions.ImageFormat:\x10IMAGE_FORMAT_RGBR\x0bimageFormat\x12\x95\x01\n\x19\x64\x65scriptor_extractor_type\x18\x41 \x01(\x0e\x32?.mediapipe.RegionFlowComputationOptions.DescriptorExtractorType:\x18\x44\x45SCRIPTOR_EXTRACTOR_ORBR\x17\x64\x65scriptorExtractorType\x12G\n\x1d\x63ompute_derivative_in_pyramid\x18\x42 \x01(\x08:\x04trueR\x1a\x63omputeDerivativeInPyramid\x1a\xfa\x01\n\x12\x44ownSampleSchedule\x12\x37\n\x16\x64ownsample_factor_360p\x18\x01 \x01(\x02:\x01\x31R\x14\x64ownsampleFactor360p\x12\x37\n\x16\x64ownsample_factor_480p\x18\x02 \x01(\x02:\x01\x31R\x14\x64ownsampleFactor480p\x12\x37\n\x16\x64ownsample_factor_720p\x18\x03 \x01(\x02:\x01\x32R\x14\x64ownsampleFactor720p\x12\x39\n\x17\x64ownsample_factor_1080p\x18\x04 \x01(\x02:\x01\x32R\x15\x64ownsampleFactor1080p\x1a\x86\x02\n\x10\x42lurScoreOptions\x12)\n\x0f\x62ox_filter_diam\x18\x01 \x01(\x05:\x01\x33R\rboxFilterDiam\x12H\n\x1drelative_cornerness_threshold\x18\x02 \x01(\x02:\x04\x30.03R\x1brelativeCornernessThreshold\x12J\n\x1d\x61\x62solute_cornerness_threshold\x18\x03 \x01(\x02:\x06\x30.0001R\x1b\x61\x62soluteCornernessThreshold\x12\x31\n\x11median_percentile\x18\x05 \x01(\x02:\x04\x30.85R\x10medianPercentile\x1a\x87\x01\n\x18VisualConsistencyOptions\x12\x35\n\x13\x63ompute_consistency\x18\x01 \x01(\x08:\x04trueR\x12\x63omputeConsistency\x12\x34\n\x14tiny_image_dimension\x18\x02 \x01(\x05:\x02\x32\x30R\x12tinyImageDimension\"]\n\x12IrlsInitialization\x12\x15\n\x11IRIS_INIT_UNKNOWN\x10\x00\x12\x15\n\x11IRIS_INIT_UNIFORM\x10\x01\x12\x19\n\x15IRIS_INIT_CONSISTENCY\x10\x02\"\xec\x01\n\x0e\x44ownsampleMode\x12\x1b\n\x17\x44OWNSAMPLE_MODE_UNKNOWN\x10\x00\x12\x18\n\x14\x44OWNSAMPLE_MODE_NONE\x10\x01\x12\x1f\n\x1b\x44OWNSAMPLE_MODE_TO_MAX_SIZE\x10\x02\x12\x1d\n\x19\x44OWNSAMPLE_MODE_BY_FACTOR\x10\x03\x12\x1f\n\x1b\x44OWNSAMPLE_MODE_BY_SCHEDULE\x10\x04\x12\x1f\n\x1b\x44OWNSAMPLE_MODE_TO_MIN_SIZE\x10\x05\x12!\n\x1d\x44OWNSAMPLE_MODE_TO_INPUT_SIZE\x10\x06\"\x90\x01\n\x0fGainCorrectMode\x12\"\n\x1eGAIN_CORRECT_MODE_DEFAULT_USER\x10\x01\x12\x1b\n\x17GAIN_CORRECT_MODE_VIDEO\x10\x02\x12\x19\n\x15GAIN_CORRECT_MODE_HDR\x10\x03\x12!\n\x1dGAIN_CORRECT_MODE_PHOTO_BURST\x10\x04\"\x9d\x01\n\x0bImageFormat\x12\x18\n\x14IMAGE_FORMAT_UNKNOWN\x10\x00\x12\x1a\n\x16IMAGE_FORMAT_GRAYSCALE\x10\x01\x12\x14\n\x10IMAGE_FORMAT_RGB\x10\x02\x12\x15\n\x11IMAGE_FORMAT_RGBA\x10\x03\x12\x14\n\x10IMAGE_FORMAT_BGR\x10\x04\x12\x15\n\x11IMAGE_FORMAT_BGRA\x10\x05\"7\n\x17\x44\x65scriptorExtractorType\x12\x1c\n\x18\x44\x45SCRIPTOR_EXTRACTOR_ORB\x10\x00*\x04\x08\x05\x10\x06*\x04\x08\x07\x10\x08*\x04\x08\x08\x10\t*\x04\x08\t\x10\n*\x04\x08\n\x10\x0b*\x04\x08\x0f\x10\x10*\x04\x08\x10\x10\x11*\x04\x08\x18\x10\x19*\x04\x08\x1d\x10\x1e*\x04\x08\x1e\x10\x1f*\x04\x08 \x10!*\x04\x08*\x10+*\x04\x08+\x10,B5Z3github.com/google/mediapipe/mediapipe/util/tracking')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.util.tracking.region_flow_computation_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z3github.com/google/mediapipe/mediapipe/util/tracking'
  _TRACKINGOPTIONS._serialized_start=116
  _TRACKINGOPTIONS._serialized_end=3220
  _TRACKINGOPTIONS_MINEIGVALEXTRACTIONSETTINGS._serialized_start=2357
  _TRACKINGOPTIONS_MINEIGVALEXTRACTIONSETTINGS._serialized_end=2518
  _TRACKINGOPTIONS_HARRISEXTRACTIONSETTINGS._serialized_start=2520
  _TRACKINGOPTIONS_HARRISEXTRACTIONSETTINGS._serialized_end=2607
  _TRACKINGOPTIONS_FASTEXTRACTIONSETTINGS._serialized_start=2609
  _TRACKINGOPTIONS_FASTEXTRACTIONSETTINGS._serialized_end=2667
  _TRACKINGOPTIONS_FLOWDIRECTION._serialized_start=2670
  _TRACKINGOPTIONS_FLOWDIRECTION._serialized_end=2804
  _TRACKINGOPTIONS_TRACKINGPOLICY._serialized_start=2807
  _TRACKINGOPTIONS_TRACKINGPOLICY._serialized_end=2952
  _TRACKINGOPTIONS_CORNEREXTRACTIONMETHOD._serialized_start=2955
  _TRACKINGOPTIONS_CORNEREXTRACTIONMETHOD._serialized_end=3131
  _TRACKINGOPTIONS_KLTTRACKERIMPLEMENTATION._serialized_start=3133
  _TRACKINGOPTIONS_KLTTRACKERIMPLEMENTATION._serialized_end=3196
  _REGIONFLOWCOMPUTATIONOPTIONS._serialized_start=3223
  _REGIONFLOWCOMPUTATIONOPTIONS._serialized_end=8607
  _REGIONFLOWCOMPUTATIONOPTIONS_DOWNSAMPLESCHEDULE._serialized_start=7178
  _REGIONFLOWCOMPUTATIONOPTIONS_DOWNSAMPLESCHEDULE._serialized_end=7428
  _REGIONFLOWCOMPUTATIONOPTIONS_BLURSCOREOPTIONS._serialized_start=7431
  _REGIONFLOWCOMPUTATIONOPTIONS_BLURSCOREOPTIONS._serialized_end=7693
  _REGIONFLOWCOMPUTATIONOPTIONS_VISUALCONSISTENCYOPTIONS._serialized_start=7696
  _REGIONFLOWCOMPUTATIONOPTIONS_VISUALCONSISTENCYOPTIONS._serialized_end=7831
  _REGIONFLOWCOMPUTATIONOPTIONS_IRLSINITIALIZATION._serialized_start=7833
  _REGIONFLOWCOMPUTATIONOPTIONS_IRLSINITIALIZATION._serialized_end=7926
  _REGIONFLOWCOMPUTATIONOPTIONS_DOWNSAMPLEMODE._serialized_start=7929
  _REGIONFLOWCOMPUTATIONOPTIONS_DOWNSAMPLEMODE._serialized_end=8165
  _REGIONFLOWCOMPUTATIONOPTIONS_GAINCORRECTMODE._serialized_start=8168
  _REGIONFLOWCOMPUTATIONOPTIONS_GAINCORRECTMODE._serialized_end=8312
  _REGIONFLOWCOMPUTATIONOPTIONS_IMAGEFORMAT._serialized_start=8315
  _REGIONFLOWCOMPUTATIONOPTIONS_IMAGEFORMAT._serialized_end=8472
  _REGIONFLOWCOMPUTATIONOPTIONS_DESCRIPTOREXTRACTORTYPE._serialized_start=8474
  _REGIONFLOWCOMPUTATIONOPTIONS_DESCRIPTOREXTRACTORTYPE._serialized_end=8529
# @@protoc_insertion_point(module_scope)