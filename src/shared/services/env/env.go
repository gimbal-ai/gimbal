/*
 * Copyright 2023- Gimlet Labs, Inc.
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

package env

import (
	"context"
	"errors"

	"github.com/spf13/viper"
)

type envContextKey struct{}

// Env is the interface that all sub-environments should implement.
type Env interface {
	JWTSigningKey() string
	ServiceName() string
	PodName() string
	Audience() string
}

// BaseEnv is the struct containing server state that is valid across multiple sessions
// for example, database connections and config information.
type BaseEnv struct {
	audience      string
	jwtSigningKey string
	serviceName   string
	podName       string
}

// New creates a new base environment use by all our services.
func New(audience, serviceName string) *BaseEnv {
	return &BaseEnv{
		audience:      audience,
		serviceName:   serviceName,
		jwtSigningKey: viper.GetString("jwt_signing_key"),
		podName:       viper.GetString("pod_name"),
	}
}

// JWTSigningKey returns the JWT key.
func (e *BaseEnv) JWTSigningKey() string {
	return e.jwtSigningKey
}

// Audience returns the audience.
func (e *BaseEnv) Audience() string {
	return e.audience
}

// ServiceName returns the name of the service.
func (e *BaseEnv) ServiceName() string {
	return e.serviceName
}

// PodName Returns the pod name if running in K8s, empty if not.
func (e *BaseEnv) PodName() string {
	return e.podName
}

// NewContext returns a new context with session context.
func NewContext(ctx context.Context, e Env) context.Context {
	return context.WithValue(ctx, envContextKey{}, e)
}

// FromContext returns a session context from the passed in Context.
func FromContext(ctx context.Context) (Env, error) {
	s, ok := ctx.Value(envContextKey{}).(Env)
	if !ok {
		return nil, errors.New("failed to get auth info from context")
	}
	return s, nil
}
