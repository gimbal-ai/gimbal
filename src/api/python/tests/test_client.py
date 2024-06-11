# Copyright © 2023- Gimlet Labs, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Gimlet Labs, Inc. and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Gimlet Labs, Inc. and its suppliers and
# may be covered by U.S. and Foreign Patents, patents in process,
# and are protected by trade secret or copyright law. Dissemination
# of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from
# Gimlet Labs, Inc.
#
# SPDX-License-Identifier: Proprietary

import concurrent.futures
import io
import struct
import sys
import tempfile
import uuid
from typing import Iterable

import gml
import gml.proto.src.common.typespb.uuid_pb2 as uuidpb
import gml.proto.src.controlplane.directory.directorypb.v1.directory_pb2 as dirpb
import gml.proto.src.controlplane.directory.directorypb.v1.directory_pb2_grpc as dirpb_grpc
import gml.proto.src.controlplane.filetransfer.ftpb.v1.ftpb_pb2 as ftpb
import gml.proto.src.controlplane.filetransfer.ftpb.v1.ftpb_pb2_grpc as ftpb_grpc
import grpc
import pytest


def uuid_to_proto(id: uuid.UUID):
    low_bits = struct.unpack("=Q", id.bytes[:8])[0]
    high_bits = struct.unpack("=Q", id.bytes[8:])[0]
    return uuidpb.UUID(
        high_bits=high_bits,
        low_bits=low_bits,
    )


def proto_to_uuid(pb: uuidpb.UUID):
    high_bytes = struct.pack("=Q", pb.high_bits)
    low_bytes = struct.pack("=Q", pb.low_bits)
    return uuid.UUID(bytes=low_bytes + high_bytes)


class FakeFileTransferServicer(ftpb_grpc.FileTransferServiceServicer):
    def __init__(self):
        self.file_ids = dict()
        self.files = dict()

    def _info_for_id(self, id):
        info = self.files[id]
        return ftpb.FileInfo(
            file_id=uuid_to_proto(id),
            status=info.get("status", None),
            name=info["name"],
            size_bytes=info.get("size_bytes", None),
            sha256sum=info.get("sha256sum", None),
        )

    def CreateFileInfo(
        self, req: ftpb.CreateFileInfoRequest, context: grpc.ServicerContext
    ):
        if req.name in self.file_ids:
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            return ftpb.CreateFileInfoResponse()
        id = uuid.uuid4()
        self.file_ids[req.name] = id
        self.files[id] = dict(name=req.name, status=ftpb.FILE_STATUS_CREATED)
        return ftpb.CreateFileInfoResponse(
            info=self._info_for_id(id),
        )

    def GetFileInfoByName(
        self, req: ftpb.GetFileInfoByNameRequest, context: grpc.ServicerContext
    ):
        if req.name not in self.file_ids:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return ftpb.GetFileInfoByNameResponse()

        id = self.file_ids[req.name]
        return ftpb.GetFileInfoByNameResponse(
            info=self._info_for_id(id),
        )

    def UploadFile(
        self, requests: Iterable[ftpb.UploadFileRequest], context: grpc.ServicerContext
    ):
        for req in requests:
            id = proto_to_uuid(req.file_id)
            if id not in self.files:
                context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
                return ftpb.UploadFileResponse()

            info = self.files[id]
            if "size_bytes" not in info:
                info["size_bytes"] = 0

            info["size_bytes"] += len(req.chunk)
            info["sha256sum"] = req.sha256sum

        self.files[id]["status"] = ftpb.FILE_STATUS_READY
        return ftpb.UploadFileResponse(
            file_id=uuid_to_proto(id), size_bytes=info["size_bytes"]
        )


ORG_NAME = "Test Org"
ORG_ID = uuid.uuid4()


class FakeOrgDirectoryServicer(dirpb_grpc.OrgDirectoryServiceServicer):
    def GetOrg(self, req: dirpb.GetOrgRequest, context: grpc.ServicerContext):
        if req.org_name != ORG_NAME:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return dirpb.GetOrgResponse()

        return dirpb.GetOrgResponse(
            org_info=dirpb.OrgInfo(id=uuid_to_proto(ORG_ID), org_name=ORG_NAME)
        )


@pytest.fixture
def gml_client():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    ftpb_grpc.add_FileTransferServiceServicer_to_server(
        FakeFileTransferServicer(), server
    )
    dirpb_grpc.add_OrgDirectoryServiceServicer_to_server(
        FakeOrgDirectoryServicer(), server
    )
    tmpdir = tempfile.mkdtemp()
    addr = "unix:" + "/".join([tmpdir, "socket"])
    server.add_insecure_port(addr)
    server.start()

    client = gml.Client(
        api_key="fake_key",
        org=ORG_NAME,
        controlplane_addr=addr,
        insecure_no_ssl=True,
    )
    yield client

    server.stop(1)
    server.wait_for_termination()


def test_upload_file(gml_client: gml.Client):
    buf = io.BytesIO(b"1234")
    file_info: ftpb.FileInfo = gml_client.upload_file("test_name", buf, chunk_size=1)

    assert file_info.status == ftpb.FILE_STATUS_READY
    assert file_info.name == "test_name"
    assert file_info.size_bytes == 4


def test_upload_file_already_exists(gml_client: gml.Client):
    buf = io.BytesIO(b"1234")
    gml_client.upload_file("test_name", buf, chunk_size=1)

    buf2 = io.BytesIO(b"5678")
    with pytest.raises(gml.client.FileAlreadyExists):
        gml_client.upload_file("test_name", buf2)


def test_upload_file_already_created(gml_client: gml.Client):
    gml_client._create_file("test_name")

    buf = io.BytesIO(b"1234")
    file_info = gml_client._upload_file_if_not_exists("test_name", buf)
    assert file_info.status == ftpb.FILE_STATUS_READY
    assert file_info.name == "test_name"
    assert file_info.size_bytes == 4


def test_get_org_id(gml_client: gml.Client):
    id = gml_client._get_org_id()
    assert proto_to_uuid(id) == ORG_ID


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
