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

package authcontext

import (
	"context"
	"time"

	"github.com/lestrrat-go/jwx/v2/jwa"
	"github.com/lestrrat-go/jwx/v2/jwk"
	"github.com/lestrrat-go/jwx/v2/jwt"
	log "github.com/sirupsen/logrus"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"gimletlabs.ai/gimlet/src/common/typespb"
	"gimletlabs.ai/gimlet/src/shared/services/utils"
)

var ErrInsufficientClaims = status.Error(codes.PermissionDenied, "insufficient claims")

type authContextKey struct{}

// AuthContext stores sessions specific information.
type AuthContext struct {
	AuthToken string
	Claims    *typespb.JWTClaims
	Path      string
	ServiceID string
}

// New creates a new session context.
func New() *AuthContext {
	return &AuthContext{}
}

// UseJWTAuth takes a token and sets claims, etc.
func (s *AuthContext) UseJWTAuth(signingKey string, tokenString string, audience string) error {
	key, err := jwk.ParseKey([]byte(signingKey))
	if err != nil {
		return err
	}
	token, err := jwt.Parse([]byte(tokenString), jwt.WithKey(jwa.HS256, key), jwt.WithAudience(audience), jwt.WithValidate(true))
	if err != nil {
		return err
	}

	s.Claims, err = utils.TokenToProto(token)
	if err != nil {
		return err
	}
	s.AuthToken = tokenString
	return nil
}

// ValidClaims returns true if the user is logged in and valid.
func (s *AuthContext) ValidClaims() bool {
	if s.Claims == nil {
		return false
	}

	if len(s.Claims.Subject) == 0 {
		return false
	}
	if s.Claims.ExpiresAt < time.Now().Unix() {
		return false
	}

	switch utils.GetClaimsType(s.Claims) {
	case utils.UserClaimType:
		return s.Claims.GetUserClaims() != nil && len(s.Claims.GetUserClaims().UserID) > 0
	case utils.ServiceClaimType:
		svcClaims := s.Claims.GetServiceClaims()
		if svcClaims == nil {
			return false
		}
		if len(svcClaims.ServiceID) == 0 || (s.ServiceID != "" && s.Claims.GetServiceClaims().ServiceID != s.ServiceID) {
			return false
		}
		return true
	case utils.DeviceClaimType:
		hasDeployKeyOrDeviceID := len(s.Claims.GetDeviceClaims().DeployKeyID) > 0 || len(s.Claims.GetDeviceClaims().DeviceID) > 0
		return len(s.Claims.GetDeviceClaims().FleetID) > 0 && hasDeployKeyOrDeviceID
	default:
	}
	return false
}

// NewContext returns a new context with session context.
func NewContext(ctx context.Context, s *AuthContext) context.Context {
	return context.WithValue(ctx, authContextKey{}, s)
}

// MustFromContext returns a session context from the passed in Context or fatals.
func MustFromContext(ctx context.Context) *AuthContext {
	s, ok := ctx.Value(authContextKey{}).(*AuthContext)
	if !ok {
		log.Fatal("failed to get auth info from context")
	}
	return s
}
