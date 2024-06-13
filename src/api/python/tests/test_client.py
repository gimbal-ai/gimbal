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
import os
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

import src.controlplane.logicalpipeline.lppb.v1.lppb_pb2 as lppb
import src.controlplane.logicalpipeline.lppb.v1.lppb_pb2_grpc as lppb_grpc
import src.controlplane.model.mpb.v1.mpb_pb2 as mpb
import src.controlplane.model.mpb.v1.mpb_pb2_grpc as mpb_grpc


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


class FakeModelServicer(mpb_grpc.ModelServiceServicer):
    def __init__(self):
        self.model_ids_by_org = dict()
        self.models = dict()

    def CreateModel(self, req: mpb.CreateModelRequest, context: grpc.ServicerContext):
        org_id = proto_to_uuid(req.org_id)
        if org_id not in self.model_ids_by_org:
            self.model_ids_by_org[org_id] = dict()
        if req.name in self.model_ids_by_org[org_id]:
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            return mpb.CreateModelResponse()

        id = uuid.uuid4()
        self.model_ids_by_org[org_id][req.name] = id
        self.models[id] = req.model_info
        return mpb.CreateModelResponse(id=uuid_to_proto(id))


class FakeLogicalPipelineServicer(lppb_grpc.LogicalPipelineServiceServicer):
    def __init__(self):
        self.pipeline_ids_by_org = dict()
        self.pipelines = dict()

    def CreateLogicalPipeline(
        self, req: lppb.CreateLogicalPipelineRequest, context: grpc.ServicerContext
    ):
        org_id = proto_to_uuid(req.org_id)
        if org_id not in self.pipeline_ids_by_org:
            self.pipeline_ids_by_org[org_id] = dict()
        if req.name in self.pipeline_ids_by_org[org_id]:
            context.set_code(grpc.StatusCode.ALREADY_EXISTS)
            return lppb.CreateLogicalPipelineResponse()

        id = uuid.uuid4()
        self.pipeline_ids_by_org[org_id][req.name] = id
        self.pipelines[id] = req.yaml
        return lppb.CreateLogicalPipelineResponse(id=uuid_to_proto(id))


@pytest.fixture
def gml_client():
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=1))
    ftpb_grpc.add_FileTransferServiceServicer_to_server(
        FakeFileTransferServicer(), server
    )
    dirpb_grpc.add_OrgDirectoryServiceServicer_to_server(
        FakeOrgDirectoryServicer(), server
    )
    mpb_grpc.add_ModelServiceServicer_to_server(FakeModelServicer(), server)
    lppb_grpc.add_LogicalPipelineServiceServicer_to_server(
        FakeLogicalPipelineServicer(), server
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
    os.removedirs(tmpdir)


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


def test_upload_pipeline(gml_client: gml.Client):
    model = gml.Model(
        "test_model",
        torch_module=None,
        input_shapes=[],
        input_dtypes=[],
        output_bbox_format=gml.model.BoundingBoxFormat(
            box_format="cxcywh", is_normalized=True
        ),
        class_labels=["class1", "class2"],
        image_preprocessing_steps=[
            gml.preprocessing.LetterboxImage(),
            gml.preprocessing.ImageToFloatTensor(),
        ],
    )
    # stub convert_to_mlir
    model.convert_to_torch_mlir = lambda *args, **kwargs: "test model contents"

    pipeline = "test pipeline contents"

    # Check that upload_pipeline doesn't fail.
    gml_client.upload_pipeline(
        name="test pipeline",
        models=[model],
        pipeline=pipeline,
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
