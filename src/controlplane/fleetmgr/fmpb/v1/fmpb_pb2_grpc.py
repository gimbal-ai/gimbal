# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from src.controlplane.fleetmgr.fmpb.v1 import fmpb_pb2 as src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2


class FleetMgrServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.CreateFleet = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/CreateFleet',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.CreateFleetRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.CreateFleetResponse.FromString,
                )
        self.GetFleet = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/GetFleet',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetResponse.FromString,
                )
        self.GetFleetByName = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/GetFleetByName',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetByNameRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetByNameResponse.FromString,
                )
        self.ListFleets = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/ListFleets',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListFleetsRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListFleetsResponse.FromString,
                )
        self.UpdateFleet = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/UpdateFleet',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateFleetRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateFleetResponse.FromString,
                )
        self.GetDefaultTags = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/GetDefaultTags',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDefaultTagsRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDefaultTagsResponse.FromString,
                )
        self.UpsertDefaultTag = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/UpsertDefaultTag',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertDefaultTagRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertDefaultTagResponse.FromString,
                )
        self.DeleteDefaultTag = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/DeleteDefaultTag',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDefaultTagRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDefaultTagResponse.FromString,
                )


class FleetMgrServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def CreateFleet(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetFleet(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetFleetByName(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListFleets(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateFleet(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDefaultTags(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpsertDefaultTag(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteDefaultTag(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FleetMgrServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'CreateFleet': grpc.unary_unary_rpc_method_handler(
                    servicer.CreateFleet,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.CreateFleetRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.CreateFleetResponse.SerializeToString,
            ),
            'GetFleet': grpc.unary_unary_rpc_method_handler(
                    servicer.GetFleet,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetResponse.SerializeToString,
            ),
            'GetFleetByName': grpc.unary_unary_rpc_method_handler(
                    servicer.GetFleetByName,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetByNameRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetByNameResponse.SerializeToString,
            ),
            'ListFleets': grpc.unary_unary_rpc_method_handler(
                    servicer.ListFleets,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListFleetsRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListFleetsResponse.SerializeToString,
            ),
            'UpdateFleet': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateFleet,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateFleetRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateFleetResponse.SerializeToString,
            ),
            'GetDefaultTags': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDefaultTags,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDefaultTagsRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDefaultTagsResponse.SerializeToString,
            ),
            'UpsertDefaultTag': grpc.unary_unary_rpc_method_handler(
                    servicer.UpsertDefaultTag,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertDefaultTagRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertDefaultTagResponse.SerializeToString,
            ),
            'DeleteDefaultTag': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteDefaultTag,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDefaultTagRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDefaultTagResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'gml.internal.controlplane.fleetmgr.v1.FleetMgrService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FleetMgrService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def CreateFleet(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/CreateFleet',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.CreateFleetRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.CreateFleetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetFleet(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/GetFleet',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetFleetByName(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/GetFleetByName',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetByNameRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetFleetByNameResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListFleets(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/ListFleets',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListFleetsRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListFleetsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateFleet(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/UpdateFleet',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateFleetRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateFleetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDefaultTags(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/GetDefaultTags',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDefaultTagsRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDefaultTagsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpsertDefaultTag(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/UpsertDefaultTag',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertDefaultTagRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertDefaultTagResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteDefaultTag(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrService/DeleteDefaultTag',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDefaultTagRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDefaultTagResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class FleetMgrEdgeServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.AssociateTagsWithDeployKey = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/AssociateTagsWithDeployKey',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.AssociateTagsWithDeployKeyRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.AssociateTagsWithDeployKeyResponse.FromString,
                )
        self.Register = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/Register',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.RegisterRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.RegisterResponse.FromString,
                )
        self.UpdateStatus = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/UpdateStatus',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateStatusRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateStatusResponse.FromString,
                )
        self.GetDevice = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/GetDevice',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDeviceRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDeviceResponse.FromString,
                )
        self.ListDevices = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/ListDevices',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListDevicesRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListDevicesResponse.FromString,
                )
        self.UpdateDevice = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/UpdateDevice',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateDeviceRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateDeviceResponse.FromString,
                )
        self.DeleteDevices = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/DeleteDevices',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDevicesRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDevicesResponse.FromString,
                )
        self.SetDeviceCapabilities = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/SetDeviceCapabilities',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.SetDeviceCapabilitiesRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.SetDeviceCapabilitiesResponse.FromString,
                )
        self.GetTags = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/GetTags',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetTagsRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetTagsResponse.FromString,
                )
        self.UpsertTag = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/UpsertTag',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertTagRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertTagResponse.FromString,
                )
        self.DeleteTag = channel.unary_unary(
                '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/DeleteTag',
                request_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteTagRequest.SerializeToString,
                response_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteTagResponse.FromString,
                )


class FleetMgrEdgeServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def AssociateTagsWithDeployKey(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def Register(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateStatus(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetDevice(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ListDevices(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpdateDevice(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteDevices(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def SetDeviceCapabilities(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetTags(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def UpsertTag(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def DeleteTag(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_FleetMgrEdgeServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'AssociateTagsWithDeployKey': grpc.unary_unary_rpc_method_handler(
                    servicer.AssociateTagsWithDeployKey,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.AssociateTagsWithDeployKeyRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.AssociateTagsWithDeployKeyResponse.SerializeToString,
            ),
            'Register': grpc.unary_unary_rpc_method_handler(
                    servicer.Register,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.RegisterRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.RegisterResponse.SerializeToString,
            ),
            'UpdateStatus': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateStatus,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateStatusRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateStatusResponse.SerializeToString,
            ),
            'GetDevice': grpc.unary_unary_rpc_method_handler(
                    servicer.GetDevice,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDeviceRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDeviceResponse.SerializeToString,
            ),
            'ListDevices': grpc.unary_unary_rpc_method_handler(
                    servicer.ListDevices,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListDevicesRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListDevicesResponse.SerializeToString,
            ),
            'UpdateDevice': grpc.unary_unary_rpc_method_handler(
                    servicer.UpdateDevice,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateDeviceRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateDeviceResponse.SerializeToString,
            ),
            'DeleteDevices': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteDevices,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDevicesRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDevicesResponse.SerializeToString,
            ),
            'SetDeviceCapabilities': grpc.unary_unary_rpc_method_handler(
                    servicer.SetDeviceCapabilities,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.SetDeviceCapabilitiesRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.SetDeviceCapabilitiesResponse.SerializeToString,
            ),
            'GetTags': grpc.unary_unary_rpc_method_handler(
                    servicer.GetTags,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetTagsRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetTagsResponse.SerializeToString,
            ),
            'UpsertTag': grpc.unary_unary_rpc_method_handler(
                    servicer.UpsertTag,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertTagRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertTagResponse.SerializeToString,
            ),
            'DeleteTag': grpc.unary_unary_rpc_method_handler(
                    servicer.DeleteTag,
                    request_deserializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteTagRequest.FromString,
                    response_serializer=src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteTagResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class FleetMgrEdgeService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def AssociateTagsWithDeployKey(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/AssociateTagsWithDeployKey',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.AssociateTagsWithDeployKeyRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.AssociateTagsWithDeployKeyResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def Register(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/Register',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.RegisterRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.RegisterResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateStatus(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/UpdateStatus',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateStatusRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateStatusResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetDevice(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/GetDevice',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDeviceRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetDeviceResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ListDevices(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/ListDevices',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListDevicesRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.ListDevicesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpdateDevice(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/UpdateDevice',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateDeviceRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpdateDeviceResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteDevices(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/DeleteDevices',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDevicesRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteDevicesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def SetDeviceCapabilities(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/SetDeviceCapabilities',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.SetDeviceCapabilitiesRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.SetDeviceCapabilitiesResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetTags(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/GetTags',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetTagsRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.GetTagsResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def UpsertTag(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/UpsertTag',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertTagRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.UpsertTagResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def DeleteTag(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/gml.internal.controlplane.fleetmgr.v1.FleetMgrEdgeService/DeleteTag',
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteTagRequest.SerializeToString,
            src_dot_controlplane_dot_fleetmgr_dot_fmpb_dot_v1_dot_fmpb__pb2.DeleteTagResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
