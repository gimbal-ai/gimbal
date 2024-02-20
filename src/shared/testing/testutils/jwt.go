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

package testutils

import (
	"crypto/rand"
	"encoding/json"
	"testing"
	"time"

	"github.com/lestrrat-go/jwx/v2/jwk"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/common/typespb"
	"gimletlabs.ai/gimlet/src/shared/services/utils"
)

// TestOrgID is a test org valid UUID.
const TestOrgID string = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"

// TestUserID is a test user valid UUID.
const TestUserID string = "7ba7b810-9dad-11d1-80b4-00c04fd430c8"

// GenerateTestClaimsWithDuration generates valid test user claims for a specified duration.
func GenerateTestClaimsWithDuration(_ *testing.T, duration time.Duration, email string) *typespb.JWTClaims {
	claims := utils.GenerateJWTForUser(TestUserID, []string{TestOrgID}, email, time.Now().Add(duration), "gml.ai", false)
	return claims
}

// GenerateTestServiceClaims generates valid test service claims for a specified duration.
func GenerateTestServiceClaims(_ *testing.T, service string) *typespb.JWTClaims {
	claims := utils.GenerateJWTForService(service, "gml.ai")
	return claims
}

// GenerateTestClaims generates valid test user claims valid for 60 minutes.
func GenerateTestClaims(t *testing.T) *typespb.JWTClaims {
	return GenerateTestClaimsWithDuration(t, time.Minute*60, "test@test.com")
}

// GenerateTestClaimsWithEmail generates valid test user claims for the given email.
func GenerateTestClaimsWithEmail(t *testing.T, email string) *typespb.JWTClaims {
	return GenerateTestClaimsWithDuration(t, time.Minute*60, email)
}

// GenerateTestDeviceClaims generates valid test device claims.
func GenerateTestDeviceClaims(_ *testing.T, deviceID string, fleetID string) *typespb.JWTClaims {
	claims := utils.GenerateJWTForDevice(deviceID, fleetID, "gml.ai")
	return claims
}

// GenerateTestJWTToken generates valid tokens for testing.
func GenerateTestJWTToken(t *testing.T, signingKey string) string {
	return GenerateTestJWTTokenWithDuration(t, signingKey, time.Minute*60)
}

// GenerateTestJWTTokenWithDuration generates valid tokens for testing with the specified duration.
func GenerateTestJWTTokenWithDuration(t *testing.T, signingKey string, timeout time.Duration) string {
	claims := GenerateTestClaimsWithDuration(t, timeout, "test@test.com")

	return SignPBClaims(t, claims, signingKey)
}

func GenerateJWTSigningKey(t *testing.T) string {
	octets := make([]byte, 256)
	_, err := rand.Read(octets)
	require.NoError(t, err)

	key, err := jwk.FromRaw(octets)
	require.NoError(t, err)

	enc, err := json.Marshal(key)
	require.NoError(t, err)

	return string(enc)
}

func GenerateAndSetJWTSigningKey(t *testing.T) {
	key := GenerateJWTSigningKey(t)
	viper.Set("jwt_signing_key", key)
}

// SignPBClaims signs our protobuf claims after converting to json.
func SignPBClaims(t *testing.T, claims *typespb.JWTClaims, signingKey string) string {
	signed, err := utils.SignJWTClaims(claims, signingKey)
	require.NoError(t, err)

	return signed
}
