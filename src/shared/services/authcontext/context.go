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
	"errors"
	"time"

	"github.com/lestrrat-go/jwx/jwa"
	"github.com/lestrrat-go/jwx/jwk"
	"github.com/lestrrat-go/jwx/jwt"

	"gimletlabs.ai/gimlet/src/shared/services/jwtpb"
	"gimletlabs.ai/gimlet/src/shared/services/utils"
)

type authContextKey struct{}

// AuthContext stores sessions specific information.
type AuthContext struct {
	AuthToken string
	Claims    *jwtpb.JWTClaims
	Path      string
}

// New creates a new sesion context.
func New() *AuthContext {
	return &AuthContext{}
}

// UseJWTAuth takes a token and sets claims, etc.
func (s *AuthContext) UseJWTAuth(signingKey string, tokenString string, audience string) error {
	key, err := jwk.New([]byte(signingKey))
	if err != nil {
		return err
	}
	token, err := jwt.Parse([]byte(tokenString), jwt.WithVerify(jwa.HS256, key), jwt.WithAudience(audience), jwt.WithValidate(true))
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
		return s.Claims.GetServiceClaims() != nil && len(s.Claims.GetServiceClaims().ServiceID) > 0
	default:
	}
	return false
}

// NewContext returns a new context with session context.
func NewContext(ctx context.Context, s *AuthContext) context.Context {
	return context.WithValue(ctx, authContextKey{}, s)
}

// FromContext returns a session context from the passed in Context.
func FromContext(ctx context.Context) (*AuthContext, error) {
	s, ok := ctx.Value(authContextKey{}).(*AuthContext)
	if !ok {
		return nil, errors.New("failed to get auth info from context")
	}
	return s, nil
}