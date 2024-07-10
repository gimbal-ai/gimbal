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

package utils_test

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/lestrrat-go/jwx/v2/jwt"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/common/typespb"
	"gimletlabs.ai/gimlet/src/shared/services/utils"
)

func getStandardClaimsPb() *typespb.JWTClaims {
	return &typespb.JWTClaims{
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
	userClaims := &typespb.UserJWTClaims{
		UserID: "user_id",
		Email:  "user@email.com",
		Authorizations: []*typespb.UserJWTClaims_AuthorizationDetails{
			{
				Scopes: []string{"org_admin", "org_reader"},
				OrgIDs: []string{"org1", "org2"},
			},
			{
				Scopes: []string{"org_admin"},
				OrgIDs: []string{"org3", "org4"},
			},
		},
	}
	p.CustomClaims = &typespb.JWTClaims_UserClaims{
		UserClaims: userClaims,
	}

	token, err := utils.ProtoToToken(p)
	require.NoError(t, err)

	claims := token.PrivateClaims()
	scopes, ok := claims["Scopes"]
	require.True(t, ok)
	userID, ok := claims["UserID"]
	require.True(t, ok)
	email, ok := claims["Email"]
	require.True(t, ok)
	authorizationsPb := []*typespb.UserJWTClaims_AuthorizationDetails{}
	authorizations, ok := claims["Authorizations"]
	require.True(t, ok)
	err = json.Unmarshal([]byte(authorizations.(string)), &authorizationsPb)
	require.NoError(t, err)

	assert.Equal(t, "user", scopes.(string))
	assert.Equal(t, "user_id", userID.(string))
	assert.Equal(t, "user@email.com", email.(string))
	assert.Equal(t, []*typespb.UserJWTClaims_AuthorizationDetails{
		{
			Scopes: []string{"org_admin", "org_reader"},
			OrgIDs: []string{"org1", "org2"},
		},
		{
			Scopes: []string{"org_admin"},
			OrgIDs: []string{"org3", "org4"},
		},
	}, authorizationsPb)
}

func TestProtoToToken_Service(t *testing.T) {
	p := getStandardClaimsPb()
	p.Scopes = []string{"service"}
	// Service claims.
	svcClaims := &typespb.ServiceJWTClaims{
		ServiceID: "service_id",
	}
	p.CustomClaims = &typespb.JWTClaims_ServiceClaims{
		ServiceClaims: svcClaims,
	}

	token, err := utils.ProtoToToken(p)
	require.NoError(t, err)

	claims := token.PrivateClaims()
	scopes, ok := claims["Scopes"]
	require.True(t, ok)
	serviceID, ok := claims["ServiceID"]
	require.True(t, ok)
	assert.Equal(t, "service", scopes.(string))
	assert.Equal(t, "service_id", serviceID.(string))
}

func TestProtoToToken_Device(t *testing.T) {
	p := getStandardClaimsPb()
	p.Scopes = []string{"device"}
	// Device claims.
	deviceClaims := &typespb.DeviceJWTClaims{
		DeviceID:    "device_id",
		FleetID:     "fleet_id",
		DeployKeyID: "deploy_key_id",
	}
	p.CustomClaims = &typespb.JWTClaims_DeviceClaims{
		DeviceClaims: deviceClaims,
	}

	token, err := utils.ProtoToToken(p)
	require.NoError(t, err)

	claims := token.PrivateClaims()
	scopes, ok := claims["Scopes"]
	require.True(t, ok)
	fleetID, ok := claims["FleetID"]
	require.True(t, ok)
	deviceID, ok := claims["DeviceID"]
	require.True(t, ok)
	deployKeyID, ok := claims["DeployKeyID"]
	require.True(t, ok)

	assert.Equal(t, "device", scopes.(string))
	assert.Equal(t, "device_id", deviceID.(string))
	assert.Equal(t, "fleet_id", fleetID.(string))
	assert.Equal(t, "deploy_key_id", deployKeyID.(string))
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
	authorizations := []*typespb.UserJWTClaims_AuthorizationDetails{
		{
			Scopes: []string{"org_admin", "org_reader"},
			OrgIDs: []string{"org1", "org2"},
		},
		{
			Scopes: []string{"org_admin"},
			OrgIDs: []string{"org3", "org4"},
		},
	}
	authJSON, _ := json.Marshal(authorizations)

	builder := getStandardClaimsBuilder().
		Claim("Scopes", "user").
		Claim("UserID", "user_id").
		Claim("OrgID", "org_id").
		Claim("Email", "user@email.com").
		Claim("IsAPIUser", false).
		Claim("Authorizations", string(authJSON))

	token, err := builder.Build()
	require.NoError(t, err)

	pb, err := utils.TokenToProto(token)
	require.NoError(t, err)
	assert.Equal(t, []string{"user"}, pb.Scopes)
	customClaims := pb.GetUserClaims()
	assert.Equal(t, "user_id", customClaims.UserID)
	assert.Equal(t, "user@email.com", customClaims.Email)
	assert.Equal(t, authorizations, customClaims.Authorizations)
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

func TestTokenToProto_Device(t *testing.T) {
	builder := getStandardClaimsBuilder().
		Claim("Scopes", "device").
		Claim("DeviceID", "device_id").
		Claim("FleetID", "fleet_id")

	token, err := builder.Build()
	require.NoError(t, err)

	pb, err := utils.TokenToProto(token)
	require.NoError(t, err)
	assert.Equal(t, []string{"device"}, pb.Scopes)
	customClaims := pb.GetDeviceClaims()
	assert.Equal(t, "device_id", customClaims.DeviceID)
	assert.Equal(t, "fleet_id", customClaims.FleetID)
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
	require.Error(t, err)
}
