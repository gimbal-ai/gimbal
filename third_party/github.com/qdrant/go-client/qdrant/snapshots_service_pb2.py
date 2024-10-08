# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: qdrant/snapshots_service.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1eqdrant/snapshots_service.proto\x12\x06qdrant\x1a\x1fgoogle/protobuf/timestamp.proto\"\x1b\n\x19\x43reateFullSnapshotRequest\"\x1a\n\x18ListFullSnapshotsRequest\"@\n\x19\x44\x65leteFullSnapshotRequest\x12#\n\rsnapshot_name\x18\x01 \x01(\tR\x0csnapshotName\"@\n\x15\x43reateSnapshotRequest\x12\'\n\x0f\x63ollection_name\x18\x01 \x01(\tR\x0e\x63ollectionName\"?\n\x14ListSnapshotsRequest\x12\'\n\x0f\x63ollection_name\x18\x01 \x01(\tR\x0e\x63ollectionName\"e\n\x15\x44\x65leteSnapshotRequest\x12\'\n\x0f\x63ollection_name\x18\x01 \x01(\tR\x0e\x63ollectionName\x12#\n\rsnapshot_name\x18\x02 \x01(\tR\x0csnapshotName\"\x9a\x01\n\x13SnapshotDescription\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\x12?\n\rcreation_time\x18\x02 \x01(\x0b\x32\x1a.google.protobuf.TimestampR\x0c\x63reationTime\x12\x12\n\x04size\x18\x03 \x01(\x03R\x04size\x12\x1a\n\x08\x63hecksum\x18\x04 \x01(\tR\x08\x63hecksum\"|\n\x16\x43reateSnapshotResponse\x12N\n\x14snapshot_description\x18\x01 \x01(\x0b\x32\x1b.qdrant.SnapshotDescriptionR\x13snapshotDescription\x12\x12\n\x04time\x18\x02 \x01(\x01R\x04time\"}\n\x15ListSnapshotsResponse\x12P\n\x15snapshot_descriptions\x18\x01 \x03(\x0b\x32\x1b.qdrant.SnapshotDescriptionR\x14snapshotDescriptions\x12\x12\n\x04time\x18\x02 \x01(\x01R\x04time\",\n\x16\x44\x65leteSnapshotResponse\x12\x12\n\x04time\x18\x01 \x01(\x01R\x04time2\xdd\x03\n\tSnapshots\x12I\n\x06\x43reate\x12\x1d.qdrant.CreateSnapshotRequest\x1a\x1e.qdrant.CreateSnapshotResponse\"\x00\x12\x45\n\x04List\x12\x1c.qdrant.ListSnapshotsRequest\x1a\x1d.qdrant.ListSnapshotsResponse\"\x00\x12I\n\x06\x44\x65lete\x12\x1d.qdrant.DeleteSnapshotRequest\x1a\x1e.qdrant.DeleteSnapshotResponse\"\x00\x12Q\n\nCreateFull\x12!.qdrant.CreateFullSnapshotRequest\x1a\x1e.qdrant.CreateSnapshotResponse\"\x00\x12M\n\x08ListFull\x12 .qdrant.ListFullSnapshotsRequest\x1a\x1d.qdrant.ListSnapshotsResponse\"\x00\x12Q\n\nDeleteFull\x12!.qdrant.DeleteFullSnapshotRequest\x1a\x1e.qdrant.DeleteSnapshotResponse\"\x00\x42$Z\"github.com/qdrant/go-client/qdrantb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'qdrant.snapshots_service_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z\"github.com/qdrant/go-client/qdrant'
  _CREATEFULLSNAPSHOTREQUEST._serialized_start=75
  _CREATEFULLSNAPSHOTREQUEST._serialized_end=102
  _LISTFULLSNAPSHOTSREQUEST._serialized_start=104
  _LISTFULLSNAPSHOTSREQUEST._serialized_end=130
  _DELETEFULLSNAPSHOTREQUEST._serialized_start=132
  _DELETEFULLSNAPSHOTREQUEST._serialized_end=196
  _CREATESNAPSHOTREQUEST._serialized_start=198
  _CREATESNAPSHOTREQUEST._serialized_end=262
  _LISTSNAPSHOTSREQUEST._serialized_start=264
  _LISTSNAPSHOTSREQUEST._serialized_end=327
  _DELETESNAPSHOTREQUEST._serialized_start=329
  _DELETESNAPSHOTREQUEST._serialized_end=430
  _SNAPSHOTDESCRIPTION._serialized_start=433
  _SNAPSHOTDESCRIPTION._serialized_end=587
  _CREATESNAPSHOTRESPONSE._serialized_start=589
  _CREATESNAPSHOTRESPONSE._serialized_end=713
  _LISTSNAPSHOTSRESPONSE._serialized_start=715
  _LISTSNAPSHOTSRESPONSE._serialized_end=840
  _DELETESNAPSHOTRESPONSE._serialized_start=842
  _DELETESNAPSHOTRESPONSE._serialized_end=886
  _SNAPSHOTS._serialized_start=889
  _SNAPSHOTS._serialized_end=1366
# @@protoc_insertion_point(module_scope)
