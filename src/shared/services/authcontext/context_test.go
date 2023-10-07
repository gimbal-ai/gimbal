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

func TestSessionCtx_UseJWTAuth(t *testing.T) {
	token := testutils.GenerateTestJWTToken(t, "signing_key")

	ctx := authcontext.New()
	err := ctx.UseJWTAuth("signing_key", token, "gml.ai")
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
			name:          "expired service claims",
			isValid:       false,
			claims:        testutils.GenerateTestServiceClaims(t, "api"),
			expiryFromNow: -1 * time.Second,
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

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			ctx := authcontext.New()
			if tc.claims != nil {
				token := testutils.SignPBClaims(t, tc.claims, "signing_key")
				err := ctx.UseJWTAuth("signing_key", token, "gml.ai")
				require.NoError(t, err)

				ctx.Claims.ExpiresAt = time.Now().Add(tc.expiryFromNow).Unix()
			}

			assert.Equal(t, tc.isValid, ctx.ValidClaims())
		})
	}
}
