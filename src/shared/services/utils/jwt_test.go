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

package utils_test

import (
	"testing"
	"time"

	"github.com/lestrrat-go/jwx/v2/jwt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/shared/services/jwtpb"
	"gimletlabs.ai/gimlet/src/shared/services/utils"
)

func getStandardClaimsPb() *jwtpb.JWTClaims {
	return &jwtpb.JWTClaims{
		Audience:  "audience",
		ExpiresAt: 100,
		JTI:       "jti",
		IssuedAt:  15,
		Issuer:    "issuer",
		NotBefore: 5,
		Subject:   "subject",
	}
}

func getStandardClaimsBuilder() *jwt.Builder {
	return jwt.NewBuilder().
		Audience([]string{"audience"}).
		Expiration(time.Unix(100, 0)).
		IssuedAt(time.Unix(15, 0)).
		Issuer("issuer").
		JwtID("jti").
		NotBefore(time.Unix(5, 0)).
		Subject("subject")
}

func TestProtoToToken_Standard(t *testing.T) {
	p := getStandardClaimsPb()

	token, err := utils.ProtoToToken(p)
	require.NoError(t, err)
	assert.Equal(t, "audience", token.Audience()[0])
	assert.Equal(t, int64(100), token.Expiration().Unix())
	assert.Equal(t, "jti", token.JwtID())
	assert.Equal(t, int64(15), token.IssuedAt().Unix())
	assert.Equal(t, "issuer", token.Issuer())
	assert.Equal(t, int64(5), token.NotBefore().Unix())
	assert.Equal(t, "subject", token.Subject())
}
func TestProtoToToken_User(t *testing.T) {
	p := getStandardClaimsPb()
	p.Scopes = []string{"user"}
	// User claims.
	userClaims := &jwtpb.UserJWTClaims{
		UserID: "user_id",
		Email:  "user@email.com",
	}
	p.CustomClaims = &jwtpb.JWTClaims_UserClaims{
		UserClaims: userClaims,
	}

	token, err := utils.ProtoToToken(p)
	require.NoError(t, err)

	assert.Equal(t, []string{"user"}, utils.GetScopes(token))
	assert.Equal(t, "user_id", utils.GetUserID(token))
	assert.Equal(t, "user@email.com", utils.GetEmail(token))
}

func TestProtoToToken_Service(t *testing.T) {
	p := getStandardClaimsPb()
	p.Scopes = []string{"service"}
	// Service claims.
	svcClaims := &jwtpb.ServiceJWTClaims{
		ServiceID: "service_id",
	}
	p.CustomClaims = &jwtpb.JWTClaims_ServiceClaims{
		ServiceClaims: svcClaims,
	}

	token, err := utils.ProtoToToken(p)
	require.NoError(t, err)

	assert.Equal(t, []string{"service"}, utils.GetScopes(token))
	assert.Equal(t, "service_id", utils.GetServiceID(token))
}

func TestTokenToProto_Standard(t *testing.T) {
	builder := getStandardClaimsBuilder()

	token, err := builder.Build()
	require.NoError(t, err)

	pb, err := utils.TokenToProto(token)
	require.NoError(t, err)
	assert.Equal(t, "audience", pb.Audience)
	assert.Equal(t, int64(100), pb.ExpiresAt)
	assert.Equal(t, "jti", pb.JTI)
	assert.Equal(t, int64(15), pb.IssuedAt)
	assert.Equal(t, "issuer", pb.Issuer)
	assert.Equal(t, int64(5), pb.NotBefore)
	assert.Equal(t, "subject", pb.Subject)
}

func TestTokenToProto_User(t *testing.T) {
	builder := getStandardClaimsBuilder().
		Claim("Scopes", "user").
		Claim("UserID", "user_id").
		Claim("OrgID", "org_id").
		Claim("Email", "user@email.com").
		Claim("IsAPIUser", false)

	token, err := builder.Build()
	require.NoError(t, err)

	pb, err := utils.TokenToProto(token)
	require.NoError(t, err)
	assert.Equal(t, []string{"user"}, pb.Scopes)
	customClaims := pb.GetUserClaims()
	assert.Equal(t, "user_id", customClaims.UserID)
	assert.Equal(t, "user@email.com", customClaims.Email)
}

func TestTokenToProto_Service(t *testing.T) {
	builder := getStandardClaimsBuilder().
		Claim("Scopes", "service").
		Claim("ServiceID", "service_id")

	token, err := builder.Build()
	require.NoError(t, err)

	pb, err := utils.TokenToProto(token)
	require.NoError(t, err)
	assert.Equal(t, []string{"service"}, pb.Scopes)
	customClaims := pb.GetServiceClaims()
	assert.Equal(t, "service_id", customClaims.ServiceID)
}

func TestTokenToProto_FailNoAudience(t *testing.T) {
	builder := jwt.NewBuilder().
		Expiration(time.Unix(100, 0)).
		IssuedAt(time.Unix(15, 0)).
		Issuer("issuer").
		JwtID("jti").
		NotBefore(time.Unix(5, 0)).
		Subject("subject")

	token, err := builder.Build()
	require.NoError(t, err)

	_, err = utils.TokenToProto(token)
	assert.Error(t, err)
}
