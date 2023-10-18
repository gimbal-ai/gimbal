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
