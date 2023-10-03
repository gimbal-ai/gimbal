/*
 * Copyright © 2023- Gimlet Labs, Inc.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Gimlet Labs, Inc. and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Gimlet Labs, Inc. and its suppliers and
 * may be covered by U.S. and Foreign Patents, patents in process,
 * and are protected by trade secret or copyright law. Dissemination
 * of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from
 * Gimlet Labs, Inc.
 *
 * SPDX-License-Identifier: Proprietary
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
