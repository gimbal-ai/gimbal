# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: src/controlplane/filetransfer/ftpb/v1/ftpb.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from gogoproto import gogo_pb2 as gogoproto_dot_gogo__pb2
from src.common.typespb import uuid_pb2 as src_dot_common_dot_typespb_dot_uuid__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n0src/controlplane/filetransfer/ftpb/v1/ftpb.proto\x12)gml.internal.controlplane.filetransfer.v1\x1a\x14gogoproto/gogo.proto\x1a\x1dsrc/common/typespb/uuid.proto\"\x80\x02\n\x08\x46ileInfo\x12?\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\x15\xe2\xde\x1f\x06\x46ileID\xf2\xde\x1f\x07\x64\x62:\"id\"R\x06\x66ileId\x12M\n\x06status\x18\x02 \x01(\x0e\x32\x35.gml.internal.controlplane.filetransfer.v1.FileStatusR\x06status\x12\x12\n\x04name\x18\x03 \x01(\tR\x04name\x12\x32\n\nsize_bytes\x18\x04 \x01(\x04\x42\x13\xf2\xde\x1f\x0f\x64\x62:\"size_bytes\"R\tsizeBytes\x12\x1c\n\tsha256sum\x18\x05 \x01(\tR\tsha256sum\"+\n\x15\x43reateFileInfoRequest\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\"a\n\x16\x43reateFileInfoResponse\x12G\n\x04info\x18\x01 \x01(\x0b\x32\x33.gml.internal.controlplane.filetransfer.v1.FileInfoR\x04info\"J\n\x12GetFileInfoRequest\x12\x34\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\"^\n\x13GetFileInfoResponse\x12G\n\x04info\x18\x01 \x01(\x0b\x32\x33.gml.internal.controlplane.filetransfer.v1.FileInfoR\x04info\".\n\x18GetFileInfoByNameRequest\x12\x12\n\x04name\x18\x01 \x01(\tR\x04name\"d\n\x19GetFileInfoByNameResponse\x12G\n\x04info\x18\x01 \x01(\x0b\x32\x33.gml.internal.controlplane.filetransfer.v1.FileInfoR\x04info\"}\n\x11UploadFileRequest\x12\x34\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\x12\x1c\n\tsha256sum\x18\x02 \x01(\tR\tsha256sum\x12\x14\n\x05\x63hunk\x18\x03 \x01(\x0cR\x05\x63hunk\"i\n\x12UploadFileResponse\x12\x34\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\x12\x1d\n\nsize_bytes\x18\x02 \x01(\x04R\tsizeBytes\"K\n\x13\x44ownloadFileRequest\x12\x34\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\",\n\x14\x44ownloadFileResponse\x12\x14\n\x05\x63hunk\x18\x01 \x01(\x0cR\x05\x63hunk\"_\n\x11\x44\x65leteFileRequest\x12\x34\n\x07\x66ile_id\x18\x01 \x01(\x0b\x32\x0f.gml.types.UUIDB\n\xe2\xde\x1f\x06\x46ileIDR\x06\x66ileId\x12\x14\n\x05purge\x18\x02 \x01(\x08R\x05purge\"\x14\n\x12\x44\x65leteFileResponse*n\n\nFileStatus\x12\x17\n\x13\x46ILE_STATUS_UNKNOWN\x10\x00\x12\x17\n\x13\x46ILE_STATUS_CREATED\x10\n\x12\x15\n\x11\x46ILE_STATUS_READY\x10\x14\x12\x17\n\x13\x46ILE_STATUS_DELETED\x10\x1e\x32\x8b\x07\n\x13\x46ileTransferService\x12\x95\x01\n\x0e\x43reateFileInfo\x12@.gml.internal.controlplane.filetransfer.v1.CreateFileInfoRequest\x1a\x41.gml.internal.controlplane.filetransfer.v1.CreateFileInfoResponse\x12\x8c\x01\n\x0bGetFileInfo\x12=.gml.internal.controlplane.filetransfer.v1.GetFileInfoRequest\x1a>.gml.internal.controlplane.filetransfer.v1.GetFileInfoResponse\x12\x9e\x01\n\x11GetFileInfoByName\x12\x43.gml.internal.controlplane.filetransfer.v1.GetFileInfoByNameRequest\x1a\x44.gml.internal.controlplane.filetransfer.v1.GetFileInfoByNameResponse\x12\x8b\x01\n\nUploadFile\x12<.gml.internal.controlplane.filetransfer.v1.UploadFileRequest\x1a=.gml.internal.controlplane.filetransfer.v1.UploadFileResponse(\x01\x12\x91\x01\n\x0c\x44ownloadFile\x12>.gml.internal.controlplane.filetransfer.v1.DownloadFileRequest\x1a?.gml.internal.controlplane.filetransfer.v1.DownloadFileResponse0\x01\x12\x89\x01\n\nDeleteFile\x12<.gml.internal.controlplane.filetransfer.v1.DeleteFileRequest\x1a=.gml.internal.controlplane.filetransfer.v1.DeleteFileResponseBAZ?gimletlabs.ai/gimlet/src/controlplane/filetransfer/ftpb/v1;ftpbb\x06proto3')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'src.controlplane.filetransfer.ftpb.v1.ftpb_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:

  DESCRIPTOR._options = None
  DESCRIPTOR._serialized_options = b'Z?gimletlabs.ai/gimlet/src/controlplane/filetransfer/ftpb/v1;ftpb'
  _FILEINFO.fields_by_name['file_id']._options = None
  _FILEINFO.fields_by_name['file_id']._serialized_options = b'\342\336\037\006FileID\362\336\037\007db:\"id\"'
  _FILEINFO.fields_by_name['size_bytes']._options = None
  _FILEINFO.fields_by_name['size_bytes']._serialized_options = b'\362\336\037\017db:\"size_bytes\"'
  _GETFILEINFOREQUEST.fields_by_name['file_id']._options = None
  _GETFILEINFOREQUEST.fields_by_name['file_id']._serialized_options = b'\342\336\037\006FileID'
  _UPLOADFILEREQUEST.fields_by_name['file_id']._options = None
  _UPLOADFILEREQUEST.fields_by_name['file_id']._serialized_options = b'\342\336\037\006FileID'
  _UPLOADFILERESPONSE.fields_by_name['file_id']._options = None
  _UPLOADFILERESPONSE.fields_by_name['file_id']._serialized_options = b'\342\336\037\006FileID'
  _DOWNLOADFILEREQUEST.fields_by_name['file_id']._options = None
  _DOWNLOADFILEREQUEST.fields_by_name['file_id']._serialized_options = b'\342\336\037\006FileID'
  _DELETEFILEREQUEST.fields_by_name['file_id']._options = None
  _DELETEFILEREQUEST.fields_by_name['file_id']._serialized_options = b'\342\336\037\006FileID'
  _FILESTATUS._serialized_start=1349
  _FILESTATUS._serialized_end=1459
  _FILEINFO._serialized_start=149
  _FILEINFO._serialized_end=405
  _CREATEFILEINFOREQUEST._serialized_start=407
  _CREATEFILEINFOREQUEST._serialized_end=450
  _CREATEFILEINFORESPONSE._serialized_start=452
  _CREATEFILEINFORESPONSE._serialized_end=549
  _GETFILEINFOREQUEST._serialized_start=551
  _GETFILEINFOREQUEST._serialized_end=625
  _GETFILEINFORESPONSE._serialized_start=627
  _GETFILEINFORESPONSE._serialized_end=721
  _GETFILEINFOBYNAMEREQUEST._serialized_start=723
  _GETFILEINFOBYNAMEREQUEST._serialized_end=769
  _GETFILEINFOBYNAMERESPONSE._serialized_start=771
  _GETFILEINFOBYNAMERESPONSE._serialized_end=871
  _UPLOADFILEREQUEST._serialized_start=873
  _UPLOADFILEREQUEST._serialized_end=998
  _UPLOADFILERESPONSE._serialized_start=1000
  _UPLOADFILERESPONSE._serialized_end=1105
  _DOWNLOADFILEREQUEST._serialized_start=1107
  _DOWNLOADFILEREQUEST._serialized_end=1182
  _DOWNLOADFILERESPONSE._serialized_start=1184
  _DOWNLOADFILERESPONSE._serialized_end=1228
  _DELETEFILEREQUEST._serialized_start=1230
  _DELETEFILEREQUEST._serialized_end=1325
  _DELETEFILERESPONSE._serialized_start=1327
  _DELETEFILERESPONSE._serialized_end=1347
  _FILETRANSFERSERVICE._serialized_start=1462
  _FILETRANSFERSERVICE._serialized_end=2369
# @@protoc_insertion_point(module_scope)
