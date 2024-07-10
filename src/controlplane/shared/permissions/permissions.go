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

package permissions

import (
	"github.com/gofrs/uuid/v5"

	"gimletlabs.ai/gimlet/src/common/typespb"
)

// HasOrgAccess returns whether or not the claims has access to the given org.
func HasOrgAccess(orgID uuid.UUID, claims *typespb.JWTClaims) bool {
	if claims.GetServiceClaims() != nil {
		return true
	}

	userClaims := claims.GetUserClaims()
	if userClaims == nil {
		return false
	}

	orgIDStr := orgID.String()
	for _, authorizations := range userClaims.GetAuthorizations() {
		for _, org := range authorizations.OrgIDs {
			if org == orgIDStr {
				return true
			}
		}
	}
	return false
}
