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

package utils

import (
	"time"

	"gimletlabs.ai/gimlet/src/shared/services/jwtpb"
)

// ClaimType represents the type of claims we allow in our system.
type ClaimType int

const (
	// UnknownClaimType is an unknown type.
	UnknownClaimType ClaimType = iota
	// UserClaimType is a claim for a user.
	UserClaimType
	// ServiceClaimType is a claim for a service.
	ServiceClaimType
)

// GetClaimsType gets the type of the given claim.
func GetClaimsType(c *jwtpb.JWTClaims) ClaimType {
	switch c.CustomClaims.(type) {
	case *jwtpb.JWTClaims_UserClaims:
		return UserClaimType
	case *jwtpb.JWTClaims_ServiceClaims:
		return ServiceClaimType
	default:
		return UnknownClaimType
	}
}

// GenerateJWTForUser creates a protobuf claims for the given user.
func GenerateJWTForUser(userID string, orgID string, email string, expiresAt time.Time, audience string, isAdmin bool) *jwtpb.JWTClaims {
	claims := jwtpb.JWTClaims{
		Subject: userID,
		// Standard claims.
		Audience:  audience,
		ExpiresAt: expiresAt.Unix(),
		IssuedAt:  time.Now().Unix(),
		Issuer:    "GML",
		Scopes:    []string{"user"},
	}

	orgScope := "org:user"
	if isAdmin {
		orgScope = "org:admin"
	}

	authorizations := make([]*jwtpb.UserJWTClaims_AuthorizationDetails, 0)
	authorizations = append(authorizations, &jwtpb.UserJWTClaims_AuthorizationDetails{
		Scopes: []string{orgScope},
		OrgIDs: []string{orgID},
	})

	claims.CustomClaims = &jwtpb.JWTClaims_UserClaims{
		UserClaims: &jwtpb.UserJWTClaims{
			Email:          email,
			UserID:         userID,
			Authorizations: authorizations,
		},
	}
	return &claims
}

// GenerateJWTForService creates a protobuf claims for the given service.
func GenerateJWTForService(serviceID string, audience string) *jwtpb.JWTClaims {
	pbClaims := jwtpb.JWTClaims{
		Audience:  audience,
		Subject:   serviceID,
		Issuer:    "GML",
		ExpiresAt: time.Now().Add(time.Minute * 10).Unix(),
		Scopes:    []string{"service"},
		CustomClaims: &jwtpb.JWTClaims_ServiceClaims{
			ServiceClaims: &jwtpb.ServiceJWTClaims{
				ServiceID: serviceID,
			},
		},
	}
	return &pbClaims
}
