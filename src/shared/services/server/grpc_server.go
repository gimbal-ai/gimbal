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

package server

import (
	"context"
	"fmt"

	grpc_middleware "github.com/grpc-ecosystem/go-grpc-middleware/v2"
	grpc_auth "github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/auth"
	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/logging"
	log "github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	// Enables gzip encoding for GRPC.
	_ "google.golang.org/grpc/encoding/gzip"
	"google.golang.org/grpc/status"

	"gimletlabs.ai/gimlet/src/shared/services/authcontext"
	"gimletlabs.ai/gimlet/src/shared/services/env"
)

var logrusEntry *log.Entry

func init() {
	logrusEntry = log.NewEntry(log.StandardLogger())
}

// InterceptorLogger adapts logrus logger to interceptor logger.
// This code is simple enough to be copied and not imported.
func InterceptorLogger(l log.FieldLogger) logging.Logger {
	return logging.LoggerFunc(func(_ context.Context, lvl logging.Level, msg string, fields ...any) {
		f := make(map[string]any, len(fields)/2)
		i := logging.Fields(fields).Iterator()
		if i.Next() {
			k, v := i.At()
			f[k] = v
		}
		l := l.WithFields(f)

		switch lvl {
		case logging.LevelDebug:
			l.Debug(msg)
		case logging.LevelInfo:
			l.Info(msg)
		case logging.LevelWarn:
			l.Warn(msg)
		case logging.LevelError:
			l.Error(msg)
		default:
			panic(fmt.Sprintf("unknown level %v", lvl))
		}
	})
}

// GRPCServerOptions are configuration options that are passed to the GRPC server.
type GRPCServerOptions struct {
	DisableAuth       map[string]bool
	AuthMiddleware    func(context.Context, env.Env) (string, error) // Currently only used by cloud api-server.
	GRPCServerOpts    []grpc.ServerOption
	DisableMiddleware bool
}

func grpcUnaryInjectSession(env env.Env) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req interface{}, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (interface{}, error) {
		sCtx := authcontext.New()
		sCtx.Path = info.FullMethod
		sCtx.ServiceID = env.ServiceName()
		return handler(authcontext.NewContext(ctx, sCtx), req)
	}
}

func grpcStreamInjectSession(env env.Env) grpc.StreamServerInterceptor {
	return func(srv interface{}, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		sCtx := authcontext.New()
		sCtx.Path = info.FullMethod
		sCtx.ServiceID = env.ServiceName()
		wrapped := grpc_middleware.WrapServerStream(stream)
		wrapped.WrappedContext = authcontext.NewContext(stream.Context(), sCtx)
		return handler(srv, wrapped)
	}
}

func createGRPCAuthFunc(env env.Env, opts *GRPCServerOptions) func(context.Context) (context.Context, error) {
	return func(ctx context.Context) (context.Context, error) {
		var err error
		var token string

		sCtx := authcontext.MustFromContext(ctx)
		if _, ok := opts.DisableAuth[sCtx.Path]; ok {
			return ctx, nil
		}

		if opts.AuthMiddleware != nil {
			token, err = opts.AuthMiddleware(ctx, env)
			if err != nil {
				return nil, status.Errorf(codes.Internal, "Auth middleware failed: %v", err)
			}
		} else {
			token, err = grpc_auth.AuthFromMD(ctx, "bearer")
			if err != nil {
				return nil, err
			}
		}

		err = sCtx.UseJWTAuth(env.JWTSigningKey(), token, env.Audience())
		if err != nil {
			return nil, status.Errorf(codes.Unauthenticated, "invalid auth token: %v", err)
		}
		return ctx, nil
	}
}

// CreateGRPCServer creates a GRPC server with default middleware for our services.
func CreateGRPCServer(env env.Env, serverOpts *GRPCServerOptions) *grpc.Server {
	logrusOpts := []logging.Option{
		logging.WithLogOnEvents(logging.StartCall, logging.FinishCall),
	}

	opts := []grpc.ServerOption{}
	if !serverOpts.DisableMiddleware {
		opts = append(opts,
			grpc.ChainUnaryInterceptor(
				grpcUnaryInjectSession(env),
				logging.UnaryServerInterceptor(InterceptorLogger(logrusEntry), logrusOpts...),
				grpc_auth.UnaryServerInterceptor(createGRPCAuthFunc(env, serverOpts)),
			),
			grpc.ChainStreamInterceptor(
				grpcStreamInjectSession(env),
				logging.StreamServerInterceptor(InterceptorLogger(logrusEntry), logrusOpts...),
				grpc_auth.StreamServerInterceptor(createGRPCAuthFunc(env, serverOpts)),
			),
		)
	}

	opts = append(opts, serverOpts.GRPCServerOpts...)

	grpcServer := grpc.NewServer(opts...)
	return grpcServer
}
