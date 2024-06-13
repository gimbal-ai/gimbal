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

import os
import uuid
from typing import BinaryIO, Optional, TextIO

import gml.proto.src.common.typespb.uuid_pb2 as uuidpb
import gml.proto.src.controlplane.directory.directorypb.v1.directory_pb2 as directorypb
import gml.proto.src.controlplane.directory.directorypb.v1.directory_pb2_grpc as directorypb_grpc
import gml.proto.src.controlplane.filetransfer.ftpb.v1.ftpb_pb2 as ftpb
import gml.proto.src.controlplane.filetransfer.ftpb.v1.ftpb_pb2_grpc as ftpb_grpc
import gml.proto.src.controlplane.logicalpipeline.lppb.v1.lppb_pb2_grpc as lppb_grpc
import grpc
from gml._utils import chunk_file, sha256sum

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
                raise FileAlreadyExists(f"A file already exists with name: {name}")
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