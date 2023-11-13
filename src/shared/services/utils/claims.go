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
	"context"
	"fmt"
	"time"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	"google.golang.org/grpc/metadata"

	"gimletlabs.ai/gimlet/src/common/typespb"
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
	// DeviceClaimType is a claim for a device.
	DeviceClaimType
)

// GetClaimsType gets the type of the given claim.
func GetClaimsType(c *typespb.JWTClaims) ClaimType {
	switch c.CustomClaims.(type) {
	case *typespb.JWTClaims_UserClaims:
		return UserClaimType
	case *typespb.JWTClaims_ServiceClaims:
		return ServiceClaimType
	case *typespb.JWTClaims_DeviceClaims:
		return DeviceClaimType
	default:
		return UnknownClaimType
	}
}

// GenerateJWTForUser creates a protobuf claims for the given user.
func GenerateJWTForUser(userID string, orgIDs []string, email string, expiresAt time.Time, audience string, isAdmin bool) *typespb.JWTClaims {
	claims := typespb.JWTClaims{
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

	authorizations := make([]*typespb.UserJWTClaims_AuthorizationDetails, 0)
	authorizations = append(authorizations, &typespb.UserJWTClaims_AuthorizationDetails{
		Scopes: []string{orgScope},
		OrgIDs: orgIDs,
	})

	claims.CustomClaims = &typespb.JWTClaims_UserClaims{
		UserClaims: &typespb.UserJWTClaims{
			Email:          email,
			UserID:         userID,
			Authorizations: authorizations,
		},
	}
	return &claims
}

// GenerateJWTForService creates a protobuf claims for the given service.
func GenerateJWTForService(serviceID string, audience string) *typespb.JWTClaims {
	pbClaims := typespb.JWTClaims{
		Audience:  audience,
		Subject:   serviceID,
		Issuer:    "GML",
		ExpiresAt: time.Now().Add(time.Minute * 10).Unix(),
		Scopes:    []string{"service"},
		CustomClaims: &typespb.JWTClaims_ServiceClaims{
			ServiceClaims: &typespb.ServiceJWTClaims{
				ServiceID: serviceID,
			},
		},
	}
	return &pbClaims
}

// GenerateJWTForDevice creates a protobuf claims for the given device.
func GenerateJWTForDevice(deviceID string, fleetID string, audience string) *typespb.JWTClaims {
	pbClaims := typespb.JWTClaims{
		Audience:  audience,
		Subject:   deviceID,
		Issuer:    "GML",
		ExpiresAt: time.Now().Add(time.Minute * 10).Unix(),
		Scopes:    []string{"device"},
		CustomClaims: &typespb.JWTClaims_DeviceClaims{
			DeviceClaims: &typespb.DeviceJWTClaims{
				DeviceID: deviceID,
				FleetID:  fleetID,
			},
		},
	}
	return &pbClaims
}

// ContextWithServiceClaims returns a context with service claims attached to bearer auth.
func ContextWithServiceClaims(ctx context.Context, serviceID string) (context.Context, error) {
	serviceClaims := GenerateJWTForService(serviceID, viper.GetString("domain_name"))
	serviceToken, err := SignJWTClaims(serviceClaims, viper.GetString("jwt_signing_key"))
	if err != nil {
		log.WithError(err).Error("Failed to sign JWT claims")
		return ctx, err
	}

	ctx = metadata.AppendToOutgoingContext(ctx, "authorization", fmt.Sprintf("bearer %s", serviceToken))

	return ctx, nil
}
