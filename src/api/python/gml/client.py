# Copyright 2023- Gimlet Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import uuid
from pathlib import Path
from typing import BinaryIO, List, Optional, TextIO, Union

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb
import gml.proto.src.common.typespb.uuid_pb2 as uuidpb
import gml.proto.src.controlplane.directory.directorypb.v1.directory_pb2 as directorypb
import gml.proto.src.controlplane.directory.directorypb.v1.directory_pb2_grpc as directorypb_grpc
import gml.proto.src.controlplane.filetransfer.ftpb.v1.ftpb_pb2 as ftpb
import gml.proto.src.controlplane.filetransfer.ftpb.v1.ftpb_pb2_grpc as ftpb_grpc
import gml.proto.src.controlplane.logicalpipeline.lppb.v1.lppb_pb2 as lppb
import gml.proto.src.controlplane.logicalpipeline.lppb.v1.lppb_pb2_grpc as lppb_grpc
import gml.proto.src.controlplane.model.mpb.v1.mpb_pb2 as mpb
import gml.proto.src.controlplane.model.mpb.v1.mpb_pb2_grpc as mpb_grpc
import grpc
from gml._utils import chunk_file, sha256sum
from gml.model import Model
from gml.pipelines import Pipeline

DEFAULT_CONTROLPLANE_ADDR = "app.gimletlabs.ai"


class _ChannelFactory:
    """
    _ChannelFactory creates grpc channels to a controlplane.
    """

    def __init__(self, controlplane_addr: str, insecure_no_ssl=False):
        self.controlplane_addr = controlplane_addr
        self.insecure_no_ssl = insecure_no_ssl

        self._channel_cache: grpc.Channel = None

    def get_grpc_channel(self) -> grpc.Channel:
        if self._channel_cache is not None:
            return self._channel_cache
        return self._create_grpc_channel()

    def _create_grpc_channel(self) -> grpc.Channel:
        if self.insecure_no_ssl:
            return grpc.insecure_channel(self.controlplane_addr)

        creds = grpc.ssl_channel_credentials()
        return grpc.secure_channel(self.controlplane_addr, creds)


class FileAlreadyExists(Exception):
    pass


class OrgNotSet(Exception):
    pass


class APIKeyNotSet(Exception):
    pass


class Client:
    """
    Client provides authorized access to a controlplane.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        controlplane_addr: Optional[str] = None,
        org: Optional[str] = None,
        insecure_no_ssl: bool = False,
    ):
        self._org_name = org
        self._api_key = api_key
        if self._api_key is None:
            self._api_key = os.getenv("GML_API_KEY")
            if self._api_key is None:
                raise APIKeyNotSet(
                    "must provide api_key explicitly or through environment variable GML_API_KEY"
                )

        self._controlplane_addr = controlplane_addr
        if self._controlplane_addr is None:
            self._controlplane_addr = os.getenv("GML_CONTROLPLANE_ADDR")
            if self._controlplane_addr is None:
                self._controlplane_addr = DEFAULT_CONTROLPLANE_ADDR

        self._channel_factory = _ChannelFactory(
            self._controlplane_addr, insecure_no_ssl=insecure_no_ssl
        )

        self._org_id_cache: Optional[uuidpb.UUID] = None
        self._fts_stub_cache: Optional[ftpb_grpc.FileTransferServiceStub] = None
        self._lps_stub_cache: Optional[lppb_grpc.LogicalPipelineServiceStub] = None
        self._ods_stub_cache: Optional[directorypb_grpc.OrgDirectoryServiceStub] = None
        self._ms_stub_cache: Optional[mpb_grpc.ModelServiceStub] = None

    def _get_request_metadata(self, idempotent=False):
        md = [("x-api-key", self._api_key)]
        if idempotent:
            md.append(("x-idempotency-key", uuid.uuid4().hex))
        return md

    def _fts_stub(self):
        if self._fts_stub_cache is None:
            self._fts_stub_cache = ftpb_grpc.FileTransferServiceStub(
                self._channel_factory.get_grpc_channel()
            )
        return self._fts_stub_cache

    def _lps_stub(self):
        if self._lps_stub_cache is None:
            self._lps_stub_cache = lppb_grpc.LogicalPipelineServiceStub(
                self._channel_factory.get_grpc_channel()
            )
        return self._lps_stub_cache

    def _ods_stub(self):
        if self._ods_stub_cache is None:
            self._ods_stub_cache = directorypb_grpc.OrgDirectoryServiceStub(
                self._channel_factory.get_grpc_channel()
            )
        return self._ods_stub_cache

    def _ms_stub(self):
        if self._ms_stub_cache is None:
            self._ms_stub_cache = mpb_grpc.ModelServiceStub(
                self._channel_factory.get_grpc_channel()
            )
        return self._ms_stub_cache

    def _get_org_id(self):
        if self._org_name is None:
            raise OrgNotSet("organization not set for method that is org specific")
        stub = self._ods_stub()
        req = directorypb.GetOrgRequest(org_name=self._org_name)
        resp: directorypb.GetOrgResponse = stub.GetOrg(
            req, metadata=self._get_request_metadata()
        )
        return resp.org_info.id

    def _org_id(self):
        if self._org_id_cache is None:
            self._org_id_cache = self._get_org_id()
        return self._org_id_cache

    def _create_file(self, name: str) -> ftpb.FileInfo:
        stub = self._fts_stub()
        try:
            req = ftpb.CreateFileInfoRequest(name=name)
            resp: ftpb.CreateFileInfoResponse = stub.CreateFileInfo(
                req, metadata=self._get_request_metadata()
            )
        except grpc.RpcError as e:
            if e.code() == grpc.StatusCode.ALREADY_EXISTS:
                raise FileAlreadyExists(
                    f"A file already exists with name: {name}"
                ) from e
            raise e
        return resp.info

    def _file_info_by_name(self, name: str) -> ftpb.GetFileInfoByNameResponse:
        stub = self._fts_stub()

        req = ftpb.GetFileInfoByNameRequest(name=name)
        return stub.GetFileInfoByName(req, metadata=self._get_request_metadata()).info

    def _upload_created_file(
        self,
        file_id: uuidpb.UUID,
        sha256: str,
        file: TextIO | BinaryIO,
        chunk_size=64 * 1024,
    ):
        def chunked_requests():
            file.seek(0)
            for chunk in chunk_file(file, chunk_size):
                req = ftpb.UploadFileRequest(
                    file_id=file_id, sha256sum=sha256, chunk=chunk
                )
                yield req

        stub = self._fts_stub()
        resp: ftpb.UploadFileResponse = stub.UploadFile(
            chunked_requests(), metadata=self._get_request_metadata()
        )
        return resp

    def upload_file(
        self,
        name: str,
        file: TextIO | BinaryIO,
        sha256: Optional[str] = None,
        chunk_size=64 * 1024,
    ) -> ftpb.FileInfo:
        file_info = self._create_file(name)

        if sha256 is None:
            sha256 = sha256sum(file)
        self._upload_created_file(file_info.file_id, sha256, file, chunk_size)
        return self._file_info_by_name(name)

    def _upload_file_if_not_exists(
        self,
        name: str,
        file: TextIO | BinaryIO,
        sha256: Optional[str] = None,
    ) -> ftpb.FileInfo:
        file_info: Optional[ftpb.FileInfo] = None
        try:
            file_info = self.upload_file(name, file, sha256)
        except FileAlreadyExists:
            file_info = self._file_info_by_name(name)

        match file_info.status:
            case ftpb.FILE_STATUS_READY:
                pass
            case ftpb.FILE_STATUS_CREATED:
                self._upload_created_file(file_info.file_id, sha256, file)
                file_info = self._file_info_by_name(name)
            case _:
                raise Exception("file status is deleted or unknown, cannot re-upload")
        return file_info

    def _create_model(self, model_info: modelexecpb.ModelInfo):
        req = mpb.CreateModelRequest(
            org_id=self._get_org_id(),
            name=model_info.name,
            model_info=model_info,
        )
        stub = self._ms_stub()
        resp = stub.CreateModel(
            req, metadata=self._get_request_metadata(idempotent=True)
        )
        return resp.id

    def create_model(self, model: Model):
        model_info = model.to_proto()
        for asset_name, file in model.collect_assets().items():
            if isinstance(file, Path) or isinstance(file, str):
                file = open(file, "rb")

            sha256 = sha256sum(file)

            upload_name = model.name
            if asset_name:
                upload_name += ":" + asset_name
            print(f"Uploading {upload_name}...")

            file_info = self._upload_file_if_not_exists(sha256, file, sha256)

            model_info.file_assets[asset_name].MergeFrom(file_info.file_id)

            file.close()

        return self._create_model(model_info)

    def upload_pipeline(
        self,
        *,
        name: str,
        models: List[Model],
        pipeline_file: Optional[Path] = None,
        pipeline: Optional[Union[str, Pipeline]] = None,
    ) -> uuidpb.UUID:
        if pipeline_file is not None:
            with open(pipeline_file, "r") as f:
                yaml = f.read()
        elif pipeline is not None:
            if isinstance(pipeline, Pipeline):
                if self._org_name is None:
                    raise ValueError("must set `org` to upload a pipeline")
                yaml = pipeline.to_yaml(
                    [model.name for model in models], self._org_name
                )
            else:
                yaml = pipeline
        else:
            raise ValueError("must specify one of 'pipeline_file' or 'pipeline'")

        for model in models:
            self.create_model(model)

        stub = self._lps_stub()
        req = lppb.CreateLogicalPipelineRequest(
            org_id=self._org_id(),
            name=name,
            yaml=yaml,
        )
        resp: lppb.CreateLogicalPipelineResponse = stub.CreateLogicalPipeline(
            req, metadata=self._get_request_metadata(idempotent=True)
        )
        return resp.id
