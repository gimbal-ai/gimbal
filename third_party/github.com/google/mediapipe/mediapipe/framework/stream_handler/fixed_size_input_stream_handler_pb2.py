# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: mediapipe/framework/stream_handler/fixed_size_input_stream_handler.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from mediapipe.framework import mediapipe_options_pb2 as mediapipe_dot_framework_dot_mediapipe__options__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\nHmediapipe/framework/stream_handler/fixed_size_input_stream_handler.proto\x12\tmediapipe\x1a+mediapipe/framework/mediapipe_options.proto\"\x92\x02\n\"FixedSizeInputStreamHandlerOptions\x12/\n\x12trigger_queue_size\x18\x01 \x01(\x05:\x01\x32R\x10triggerQueueSize\x12-\n\x11target_queue_size\x18\x02 \x01(\x05:\x01\x31R\x0ftargetQueueSize\x12+\n\x0e\x66ixed_min_size\x18\x03 \x01(\x08:\x05\x66\x61lseR\x0c\x66ixedMinSize2_\n\x03\x65xt\x12\x1b.mediapipe.MediaPipeOptions\x18\xbf\xe9\xfa; \x01(\x0b\x32-.mediapipe.FixedSizeInputStreamHandlerOptionsR\x03\x65xtB@Z>github.com/google/mediapipe/mediapipe/framework/stream_handler')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'mediapipe.framework.stream_handler.fixed_size_input_stream_handler_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  mediapipe_dot_framework_dot_mediapipe__options__pb2.MediaPipeOptions.RegisterExtension(_FIXEDSIZEINPUTSTREAMHANDLEROPTIONS.extensions_by_name['ext'])

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z>github.com/google/mediapipe/mediapipe/framework/stream_handler'
  _FIXEDSIZEINPUTSTREAMHANDLEROPTIONS._serialized_start=133
  _FIXEDSIZEINPUTSTREAMHANDLEROPTIONS._serialized_end=407
# @@protoc_insertion_point(module_scope)
