# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from qdrant import qdrant_pb2 as qdrant_dot_qdrant__pb2


class QdrantStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.HealthCheck = channel.unary_unary(
                '/qdrant.Qdrant/HealthCheck',
                request_serializer=qdrant_dot_qdrant__pb2.HealthCheckRequest.SerializeToString,
                response_deserializer=qdrant_dot_qdrant__pb2.HealthCheckReply.FromString,
                )


class QdrantServicer(object):
    """Missing associated documentation comment in .proto file."""

    def HealthCheck(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_QdrantServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'HealthCheck': grpc.unary_unary_rpc_method_handler(
                    servicer.HealthCheck,
                    request_deserializer=qdrant_dot_qdrant__pb2.HealthCheckRequest.FromString,
                    response_serializer=qdrant_dot_qdrant__pb2.HealthCheckReply.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'qdrant.Qdrant', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Qdrant(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def HealthCheck(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/qdrant.Qdrant/HealthCheck',
            qdrant_dot_qdrant__pb2.HealthCheckRequest.SerializeToString,
            qdrant_dot_qdrant__pb2.HealthCheckReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
