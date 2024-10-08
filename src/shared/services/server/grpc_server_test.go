/*
 * Copyright 2018- The Pixie Authors.
 * Modifications Copyright 2023- Gimlet Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

package server_test

import (
	"context"
	"fmt"
	"net"
	"testing"

	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"golang.org/x/sync/errgroup"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/grpc/status"
	"google.golang.org/grpc/test/bufconn"

	"gimletlabs.ai/gimlet/src/shared/services/env"
	"gimletlabs.ai/gimlet/src/shared/services/server"
	ping "gimletlabs.ai/gimlet/src/shared/testing/testpb"
	"gimletlabs.ai/gimlet/src/shared/testing/testutils"
)

const bufSize = 1024 * 1024

func init() {
	// Test will fail with SSL enabled since we don't expose certs to the tests.
	viper.Set("disable_ssl", true)
}

type testserver struct{}

func (*testserver) Ping(_ context.Context, _ *ping.PingRequest) (*ping.PingResponse, error) {
	return &ping.PingResponse{Reply: "test reply"}, nil
}

func (*testserver) PingServerStream(_ *ping.PingServerStreamRequest, srv ping.PingService_PingServerStreamServer) error {
	err := srv.Send(&ping.PingServerStreamResponse{Reply: "test reply"})
	if err != nil {
		return err
	}
	return nil
}

func (*testserver) PingClientStream(srv ping.PingService_PingClientStreamServer) error {
	msg, err := srv.Recv()
	if err != nil {
		return err
	}
	if msg == nil {
		return fmt.Errorf("got a nil message")
	}
	err = srv.SendAndClose(&ping.PingClientStreamResponse{Reply: "test reply"})
	if err != nil {
		return err
	}
	return nil
}

func startTestGRPCServer(t *testing.T, o *server.GRPCServerOptions) *bufconn.Listener {
	opts := o
	if opts == nil {
		opts = &server.GRPCServerOptions{}
	}

	s := server.CreateGRPCServer(env.New("gml.ai", "test-service"), opts)

	ping.RegisterPingServiceServer(s, &testserver{})
	lis := bufconn.Listen(bufSize)

	eg := errgroup.Group{}
	eg.Go(func() error { return s.Serve(lis) })

	t.Cleanup(func() {
		s.GracefulStop()

		err := eg.Wait()
		if err != nil {
			t.Fatalf("failed to start server: %v", err)
		}
	})
	return lis
}

func createDialer(lis *bufconn.Listener) func(ctx context.Context, url string) (net.Conn, error) {
	return func(_ context.Context, _ string) (net.Conn, error) {
		return lis.Dial()
	}
}

func makeTestRequest(ctx context.Context, t *testing.T, lis *bufconn.Listener) (*ping.PingResponse, error) {
	grpc.WithContextDialer(createDialer(lis))
	conn, err := grpc.NewClient("passthrough://bufnet", grpc.WithContextDialer(createDialer(lis)), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := ping.NewPingServiceClient(conn)
	return c.Ping(ctx, &ping.PingRequest{Req: "hello"})
}

func makeTestClientStreamRequest(ctx context.Context, t *testing.T, lis *bufconn.Listener) (*ping.PingClientStreamResponse, error) {
	conn, err := grpc.NewClient("passthrough://bufnet", grpc.WithContextDialer(createDialer(lis)), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := ping.NewPingServiceClient(conn)

	stream, err := c.PingClientStream(ctx)
	if err != nil {
		t.Fatalf("Could not create stream")
	}

	err = stream.Send(&ping.PingClientStreamRequest{Req: "hello"})
	if err != nil {
		t.Fatalf("Could not send on stream")
	}

	return stream.CloseAndRecv()
}

func makeTestServerStreamRequest(ctx context.Context, t *testing.T, lis *bufconn.Listener) (*ping.PingServerStreamResponse, error) {
	conn, err := grpc.NewClient("passthrough://bufnet", grpc.WithContextDialer(createDialer(lis)), grpc.WithTransportCredentials(insecure.NewCredentials()))
	if err != nil {
		t.Fatalf("did not connect: %v", err)
	}
	defer conn.Close()
	c := ping.NewPingServiceClient(conn)

	stream, err := c.PingServerStream(ctx, &ping.PingServerStreamRequest{Req: "hello"})
	if err != nil {
		t.Fatalf("Could not create stream")
	}

	return stream.Recv()
}

func TestGrpcServerUnary(t *testing.T) {
	jwtkey := testutils.GenerateJWTSigningKey(t)
	viper.Set("jwt_signing_key", jwtkey)

	tests := []struct {
		name         string
		signingKey   string
		clientStream bool
		serverStream bool
		expectError  bool
		serverOpts   *server.GRPCServerOptions
	}{
		{
			name:        "success - unary",
			signingKey:  jwtkey,
			expectError: false,
		},
		{
			name:        "bad token - unary",
			signingKey:  "{\"k\": \"bad\", \"kty\": \"oct\"}",
			expectError: true,
		},
		{
			name:        "unauthenticated - unary",
			signingKey:  "",
			expectError: true,
		},
		{
			name:         "success - client stream",
			signingKey:   jwtkey,
			expectError:  false,
			clientStream: true,
		},
		{
			name:         "bad token - client stream",
			signingKey:   "{\"k\": \"bad\", \"kty\": \"oct\"}",
			expectError:  true,
			clientStream: true,
		},
		{
			name:         "unauthenticated - client stream",
			signingKey:   "",
			expectError:  true,
			clientStream: true,
		},
		{
			name:         "success - server stream",
			signingKey:   jwtkey,
			expectError:  false,
			serverStream: true,
		},
		{
			name:         "bad token - server stream",
			signingKey:   "{\"k\": \"bad\", \"kty\": \"oct\"}",
			expectError:  true,
			serverStream: true,
		},
		{
			name:         "unauthenticated - server stream",
			signingKey:   "",
			expectError:  true,
			serverStream: true,
		},
		{
			name:         "disable auth - unary",
			signingKey:   "",
			expectError:  false,
			clientStream: false,
			serverOpts: &server.GRPCServerOptions{
				DisableAuth: map[string]bool{
					"/gml.testing.PingService/Ping": true,
				},
			},
		},
		{
			name:         "authmiddleware",
			signingKey:   "",
			expectError:  false,
			clientStream: false,
			serverOpts: &server.GRPCServerOptions{
				AuthMiddleware: func(context.Context, env.Env) (string, error) {
					return testutils.GenerateTestJWTToken(t, jwtkey), nil
				},
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			lis := startTestGRPCServer(t, test.serverOpts)

			var ctx context.Context
			if test.signingKey != "" {
				token := testutils.GenerateTestJWTToken(t, test.signingKey)
				ctx = metadata.AppendToOutgoingContext(context.Background(),
					"authorization", "bearer "+token)
			} else {
				ctx = context.Background()
			}

			var resp *ping.PingResponse
			var err error

			switch {
			case test.clientStream:
				var respint *ping.PingClientStreamResponse
				respint, err = makeTestClientStreamRequest(ctx, t, lis)
				if err == nil {
					resp = &ping.PingResponse{
						Reply: respint.Reply,
					}
				}
			case test.serverStream:
				var respint *ping.PingServerStreamResponse
				respint, err = makeTestServerStreamRequest(ctx, t, lis)
				if err == nil {
					resp = &ping.PingResponse{
						Reply: respint.Reply,
					}
				}
			default:
				resp, err = makeTestRequest(ctx, t, lis)
			}
			if test.expectError {
				require.Error(t, err)
				stat, ok := status.FromError(err)
				assert.True(t, ok)
				assert.Equal(t, codes.Unauthenticated, stat.Code())
				assert.Nil(t, resp)
			} else {
				require.NoError(t, err)
				assert.NotNil(t, resp)
				assert.Equal(t, "test reply", resp.Reply)
			}
		})
	}
}
