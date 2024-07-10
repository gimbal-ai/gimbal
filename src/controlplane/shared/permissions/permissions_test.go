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
