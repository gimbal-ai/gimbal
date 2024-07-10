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
func GenerateJWTForDevice(deviceID string, fleetID string, deployKeyID string, audience string) *typespb.JWTClaims {
	subject := deviceID
	if deviceID == "" {
		subject = deployKeyID
	}
	pbClaims := typespb.JWTClaims{
		Audience:  audience,
		Subject:   subject,
		Issuer:    "GML",
		ExpiresAt: time.Now().Add(time.Minute * 10).Unix(),
		Scopes:    []string{"device"},
		CustomClaims: &typespb.JWTClaims_DeviceClaims{
			DeviceClaims: &typespb.DeviceJWTClaims{
				DeviceID:    deviceID,
				FleetID:     fleetID,
				DeployKeyID: deployKeyID,
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

	sCtx := metadata.AppendToOutgoingContext(ctx, "authorization", fmt.Sprintf("bearer %s", serviceToken))

	return sCtx, nil
}
