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

package permissions_test

import (
	"testing"
	"time"

	"github.com/gofrs/uuid/v5"
	"github.com/stretchr/testify/assert"

	"gimletlabs.ai/gimlet/src/common/typespb"
	"gimletlabs.ai/gimlet/src/controlplane/shared/permissions"
	"gimletlabs.ai/gimlet/src/shared/testing/testutils"
)

func TestHasOrgAccess(t *testing.T) {
	tests := []struct {
		name             string
		claims           *typespb.JWTClaims
		shouldHaveAccess bool
		orgID            string
	}{
		{
			name:             "serviceClaims",
			claims:           testutils.GenerateTestServiceClaims(t, "test-service"),
			shouldHaveAccess: true,
			orgID:            "7ba7b810-9dad-11d1-80b4-00c04fd430c8",
		},
		{
			name:             "userClaims no permission",
			claims:           testutils.GenerateTestClaimsWithDuration(t, 1*time.Hour, "user@gml.ai"),
			shouldHaveAccess: false,
			orgID:            "7ba7b810-9dad-11d1-80b4-00c04fd430c8",
		},
		{
			name:             "userClaims with permission",
			claims:           testutils.GenerateTestClaimsWithDuration(t, 1*time.Hour, "user@gml.ai"),
			shouldHaveAccess: true,
			orgID:            "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			hasAccess := permissions.HasOrgAccess(uuid.FromStringOrNil(test.orgID), test.claims)
			assert.Equal(t, test.shouldHaveAccess, hasAccess)
		})
	}
}
