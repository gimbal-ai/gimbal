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

	grpcmw "github.com/grpc-ecosystem/go-grpc-middleware/v2"
	grpcauth "github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/auth"
	"github.com/grpc-ecosystem/go-grpc-middleware/v2/interceptors/logging"
	log "github.com/sirupsen/logrus"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	_ "google.golang.org/grpc/encoding/gzip" // Enables gzip encoding for GRPC.
	"google.golang.org/grpc/status"

	"gimletlabs.ai/gimlet/src/shared/services/authcontext"
	"gimletlabs.ai/gimlet/src/shared/services/env"
)

var logrusEntry *log.Entry

func init() {
	logrusEntry = log.NewEntry(log.StandardLogger())
}

func interceptorToLogrusLevel(l logging.Level) log.Level {
	switch l {
	case logging.LevelDebug:
		return log.DebugLevel
	case logging.LevelInfo:
		return log.InfoLevel
	case logging.LevelWarn:
		return log.WarnLevel
	case logging.LevelError:
		return log.TraceLevel
	}
	return log.FatalLevel
}

// InterceptorLogger adapts logrus logger to interceptor logger.
// This code is simple enough to be copied and not imported.
func InterceptorLogger(l log.FieldLogger) logging.Logger {
	return logging.LoggerFunc(func(_ context.Context, lvl logging.Level, msg string, fields ...any) {
		f := make(map[string]any, len(fields)/2)
		i := logging.Fields(fields).Iterator()
		for i.Next() {
			k, v := i.At()
			f[k] = v
		}
		l.WithFields(f).Log(interceptorToLogrusLevel(lvl), msg)
	})
}

// GRPCServerOptions are configuration options that are passed to the GRPC server.
type GRPCServerOptions struct {
	DisableAuth       map[string]bool
	AuthMiddleware    func(context.Context, env.Env) (string, error) // Currently only used by cloud api-server.
	GRPCServerOpts    []grpc.ServerOption
	DisableMiddleware bool
}

func grpcUnaryInjectSession(e env.Env) grpc.UnaryServerInterceptor {
	return func(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
		sCtx := authcontext.New()
		sCtx.Path = info.FullMethod
		sCtx.ServiceID = e.ServiceName()
		return handler(authcontext.NewContext(ctx, sCtx), req)
	}
}

func grpcStreamInjectSession(e env.Env) grpc.StreamServerInterceptor {
	return func(srv any, stream grpc.ServerStream, info *grpc.StreamServerInfo, handler grpc.StreamHandler) error {
		sCtx := authcontext.New()
		sCtx.Path = info.FullMethod
		sCtx.ServiceID = e.ServiceName()
		wrapped := grpcmw.WrapServerStream(stream)
		wrapped.WrappedContext = authcontext.NewContext(stream.Context(), sCtx)
		return handler(srv, wrapped)
	}
}

func createGRPCAuthFunc(e env.Env, opts *GRPCServerOptions) func(context.Context) (context.Context, error) {
	return func(ctx context.Context) (context.Context, error) {
		var err error
		var token string

		sCtx := authcontext.MustFromContext(ctx)
		if _, ok := opts.DisableAuth[sCtx.Path]; ok {
			return ctx, nil
		}

		if opts.AuthMiddleware != nil {
			token, err = opts.AuthMiddleware(ctx, e)
			if err != nil {
				return nil, status.Errorf(codes.Internal, "Auth middleware failed: %v", err)
			}
		} else {
			token, err = grpcauth.AuthFromMD(ctx, "bearer")
			if err != nil {
				return nil, err
			}
		}

		err = sCtx.UseJWTAuth(e.JWTSigningKey(), token, e.Audience())
		if err != nil {
			return nil, status.Errorf(codes.Unauthenticated, "invalid auth token: %v", err)
		}
		return ctx, nil
	}
}

// CreateGRPCServer creates a GRPC server with default middleware for our services.
func CreateGRPCServer(e env.Env, serverOpts *GRPCServerOptions) *grpc.Server {
	logrusOpts := []logging.Option{
		logging.WithLogOnEvents(logging.StartCall, logging.FinishCall),
	}

	var opts []grpc.ServerOption
	if !serverOpts.DisableMiddleware {
		opts = append(opts,
			grpc.ChainUnaryInterceptor(
				grpcUnaryInjectSession(e),
				logging.UnaryServerInterceptor(InterceptorLogger(logrusEntry), logrusOpts...),
				grpcauth.UnaryServerInterceptor(createGRPCAuthFunc(e, serverOpts)),
			),
			grpc.ChainStreamInterceptor(
				grpcStreamInjectSession(e),
				logging.StreamServerInterceptor(InterceptorLogger(logrusEntry), logrusOpts...),
				grpcauth.StreamServerInterceptor(createGRPCAuthFunc(e, serverOpts)),
			),
		)
	}

	opts = append(opts, serverOpts.GRPCServerOpts...)

	grpcServer := grpc.NewServer(opts...)
	return grpcServer
}
