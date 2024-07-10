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

package authcontext_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/common/typespb"
	"gimletlabs.ai/gimlet/src/shared/services/authcontext"
	"gimletlabs.ai/gimlet/src/shared/testing/testutils"
)

const (
	testFleetID   = "8ba7b810-9dad-11d1-80b4-00c04fd430c8"
	testDeviceID  = "1ba7b810-9dad-11d1-80b4-00c04fd430c8"
	testDeployKey = "7ba7b810-9dad-11d1-80b4-00c04fd43123"
)

func TestSessionCtx_UseJWTAuth(t *testing.T) {
	key := testutils.GenerateJWTSigningKey(t)
	token := testutils.GenerateTestJWTToken(t, key)

	ctx := authcontext.New()
	err := ctx.UseJWTAuth(key, token, "gml.ai")
	require.NoError(t, err)

	assert.Equal(t, testutils.TestUserID, ctx.Claims.Subject)
	assert.Equal(t, "test@test.com", ctx.Claims.GetUserClaims().Email)
}

func TestSessionCtx_ValidClaims(t *testing.T) {
	tests := []struct {
		name          string
		expiryFromNow time.Duration
		claims        *typespb.JWTClaims
		isValid       bool
	}{
		{
			name:    "no claims",
			isValid: false,
		},
		{
			name:          "valid user claims",
			isValid:       true,
			claims:        testutils.GenerateTestClaimsWithDuration(t, time.Minute*60, "test@test.com"),
			expiryFromNow: time.Minute * 60,
		},
		{
			name:          "expired user claims",
			isValid:       false,
			claims:        testutils.GenerateTestClaimsWithDuration(t, time.Minute*60, "test@test.com"),
			expiryFromNow: -1 * time.Second,
		},
		{
			name:          "valid service claims",
			isValid:       true,
			claims:        testutils.GenerateTestServiceClaims(t, "api"),
			expiryFromNow: time.Minute * 60,
		},
		{
			name:          "service claim with wrong aud",
			isValid:       false,
			claims:        testutils.GenerateTestServiceClaims(t, "directory"),
			expiryFromNow: time.Minute * 60,
		},
		{
			name:          "expired service claims",
			isValid:       false,
			claims:        testutils.GenerateTestServiceClaims(t, "api"),
			expiryFromNow: -1 * time.Second,
		},
		{
			name:          "valid device claims with deploy key",
			isValid:       true,
			claims:        testutils.GenerateTestDeviceClaims(t, "", testFleetID, testDeployKey),
			expiryFromNow: time.Minute * 60,
		},
		{
			name:          "valid device claims with device ID",
			isValid:       true,
			claims:        testutils.GenerateTestDeviceClaims(t, testDeviceID, testFleetID, ""),
			expiryFromNow: time.Minute * 60,
		},
		{
			name:          "invalid device claims is missing fleet ID",
			isValid:       false,
			claims:        testutils.GenerateTestDeviceClaims(t, "", "", testDeployKey),
			expiryFromNow: time.Minute * 60,
		},
		{
			name:          "invalid device claims is missing one of device ID or deploy key",
			isValid:       false,
			claims:        testutils.GenerateTestDeviceClaims(t, "", testFleetID, ""),
			expiryFromNow: time.Minute * 60,
		},
		{
			name:    "claims with no type",
			isValid: false,
			claims: &typespb.JWTClaims{
				Subject:   "test subject",
				Audience:  "gml.ai",
				IssuedAt:  time.Now().Unix(),
				ExpiresAt: time.Now().Add(time.Minute * 10).Unix(),
			},
			expiryFromNow: time.Minute * 60,
		},
	}

	key := testutils.GenerateJWTSigningKey(t)
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := authcontext.New()
			ctx.ServiceID = "api"
			if tc.claims != nil {
				token := testutils.SignPBClaims(t, tc.claims, key)
				err := ctx.UseJWTAuth(key, token, "gml.ai")
				require.NoError(t, err)

				ctx.Claims.ExpiresAt = time.Now().Add(tc.expiryFromNow).Unix()
			}

			assert.Equal(t, tc.isValid, ctx.ValidClaims())
		})
	}
}
