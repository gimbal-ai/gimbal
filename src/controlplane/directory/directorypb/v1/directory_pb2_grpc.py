# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from src.controlplane.directory.directorypb.v1 import directory_pb2 as src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2


class UserDirectoryServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GetUser = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.UserDirectoryService/GetUser',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetUserRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetUserResponse.FromString,
                )
        self.UpdateUser = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.UserDirectoryService/UpdateUser',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpdateUserRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpdateUserResponse.FromString,
                )
        self.DeleteUser = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.UserDirectoryService/DeleteUser',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteUserRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteUserResponse.FromString,
                )
        self.UpsertUser = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.UserDirectoryService/UpsertUser',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpsertUserRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpsertUserResponse.FromString,
                )


class UserDirectoryServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GetUser(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateUser(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteUser(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpsertUser(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_UserDirectoryServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GetUser': grpc.unary_unary_rpc_method_handler(
                    servicer.GetUser,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetUserRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetUserResponse.SerializeToString,
            ),
            'UpdateUser': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateUser,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpdateUserRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpdateUserResponse.SerializeToString,
            ),
            'DeleteUser': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteUser,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteUserRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteUserResponse.SerializeToString,
            ),
            'UpsertUser': grpc.unary_unary_rpc_method_handler(
                    servicer.UpsertUser,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpsertUserRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpsertUserResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'gml.internal.controlplane.directory.v1.UserDirectoryService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class UserDirectoryService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GetUser(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.UserDirectoryService/GetUser',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetUserRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetUserResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateUser(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.UserDirectoryService/UpdateUser',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpdateUserRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpdateUserResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteUser(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.UserDirectoryService/DeleteUser',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteUserRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteUserResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpsertUser(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.UserDirectoryService/UpsertUser',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpsertUserRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.UpsertUserResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class OrgDirectoryServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CreateOrg = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.OrgDirectoryService/CreateOrg',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.CreateOrgRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.CreateOrgResponse.FromString,
                )
        self.GetOrg = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.OrgDirectoryService/GetOrg',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetOrgRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetOrgResponse.FromString,
                )
        self.DeleteOrg = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.OrgDirectoryService/DeleteOrg',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteOrgRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteOrgResponse.FromString,
                )


class OrgDirectoryServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CreateOrg(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetOrg(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteOrg(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_OrgDirectoryServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CreateOrg': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateOrg,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.CreateOrgRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.CreateOrgResponse.SerializeToString,
            ),
            'GetOrg': grpc.unary_unary_rpc_method_handler(
                    servicer.GetOrg,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetOrgRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetOrgResponse.SerializeToString,
            ),
            'DeleteOrg': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteOrg,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteOrgRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteOrgResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'gml.internal.controlplane.directory.v1.OrgDirectoryService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class OrgDirectoryService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CreateOrg(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.OrgDirectoryService/CreateOrg',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.CreateOrgRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.CreateOrgResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetOrg(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.OrgDirectoryService/GetOrg',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetOrgRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GetOrgResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteOrg(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.OrgDirectoryService/DeleteOrg',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteOrgRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.DeleteOrgResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class OrgUserManagementServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GrantUserScopes = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.OrgUserManagementService/GrantUserScopes',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GrantUserScopesRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GrantUserScopesResponse.FromString,
                )
        self.RevokeUserScopes = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.OrgUserManagementService/RevokeUserScopes',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.RevokeUserScopesRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.RevokeUserScopesResponse.FromString,
                )
        self.ListOrgs = channel.unary_unary(
                '/gml.internal.controlplane.directory.v1.OrgUserManagementService/ListOrgs',
                request_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.ListOrgsRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.ListOrgsResponse.FromString,
                )


class OrgUserManagementServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GrantUserScopes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RevokeUserScopes(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListOrgs(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_OrgUserManagementServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GrantUserScopes': grpc.unary_unary_rpc_method_handler(
                    servicer.GrantUserScopes,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GrantUserScopesRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GrantUserScopesResponse.SerializeToString,
            ),
            'RevokeUserScopes': grpc.unary_unary_rpc_method_handler(
                    servicer.RevokeUserScopes,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.RevokeUserScopesRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.RevokeUserScopesResponse.SerializeToString,
            ),
            'ListOrgs': grpc.unary_unary_rpc_method_handler(
                    servicer.ListOrgs,
                    request_deserializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.ListOrgsRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.ListOrgsResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'gml.internal.controlplane.directory.v1.OrgUserManagementService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class OrgUserManagementService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GrantUserScopes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.OrgUserManagementService/GrantUserScopes',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GrantUserScopesRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.GrantUserScopesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RevokeUserScopes(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.OrgUserManagementService/RevokeUserScopes',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.RevokeUserScopesRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.RevokeUserScopesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListOrgs(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.directory.v1.OrgUserManagementService/ListOrgs',
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.ListOrgsRequest.SerializeToString,
            src_dot_controlplane_dot_directory_dot_directorypb_dot_v1_dot_directory__pb2.ListOrgsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)