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

package authcontext

import (
	"context"
	"time"

	"github.com/lestrrat-go/jwx/v2/jwa"
	"github.com/lestrrat-go/jwx/v2/jwk"
	"github.com/lestrrat-go/jwx/v2/jwt"
	log "github.com/sirupsen/logrus"

	"gimletlabs.ai/gimlet/src/common/typespb"
	"gimletlabs.ai/gimlet/src/shared/services/utils"
)

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
	key, err := jwk.FromRaw([]byte(signingKey))
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
		return true
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
