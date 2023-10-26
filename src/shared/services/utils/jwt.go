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
	"errors"
	"strings"
	"time"

	"github.com/lestrrat-go/jwx/v2/jwa"
	"github.com/lestrrat-go/jwx/v2/jwk"
	"github.com/lestrrat-go/jwx/v2/jwt"
	log "github.com/sirupsen/logrus"

	"gimletlabs.ai/gimlet/src/common/typespb"
)

func init() {
	// Flatten the audience array to a single string value to maintain
	// backward compatibility.
	jwt.Settings(jwt.WithFlattenAudience(true))
}

// ProtoToToken maps protobuf claims to map claims.
func ProtoToToken(pb *typespb.JWTClaims) (jwt.Token, error) {
	builder := jwt.NewBuilder()

	// Standard claims.
	builder.
		Audience([]string{pb.Audience}).
		Expiration(time.Unix(pb.ExpiresAt, 0)).
		IssuedAt(time.Unix(pb.IssuedAt, 0)).
		Issuer(pb.Issuer).
		JwtID(pb.JTI).
		NotBefore(time.Unix(pb.NotBefore, 0)).
		Subject(pb.Subject)

	// Custom claims.
	builder.
		Claim("Scopes", strings.Join(pb.Scopes, ","))

	switch m := pb.CustomClaims.(type) {
	case *typespb.JWTClaims_UserClaims:
		builder.
			Claim("UserID", m.UserClaims.UserID).
			Claim("Email", m.UserClaims.Email)
	case *typespb.JWTClaims_ServiceClaims:
		builder.Claim("ServiceID", m.ServiceClaims.ServiceID)
	case *typespb.JWTClaims_DeviceClaims:
		builder.Claim("DeviceID", m.DeviceClaims.DeviceID)
		builder.Claim("FleetID", m.DeviceClaims.FleetID)
	default:
		log.WithField("type", m).Error("Could not find claims type")
	}

	return builder.Build()
}

// TokenToProto takes a Token and converts it to a protobuf.
func TokenToProto(token jwt.Token) (*typespb.JWTClaims, error) {
	p := &typespb.JWTClaims{}

	// Standard claims.
	if len(token.Audience()) == 0 {
		return nil, errors.New("JWT has no audience")
	}
	p.Audience = token.Audience()[0]
	p.ExpiresAt = token.Expiration().Unix()
	p.IssuedAt = token.IssuedAt().Unix()
	p.Issuer = token.Issuer()
	p.JTI = token.JwtID()
	p.NotBefore = token.NotBefore().Unix()
	p.Subject = token.Subject()

	// Custom claims.
	p.Scopes = GetScopes(token)
	switch {
	case HasUserClaims(token):
		p.CustomClaims = &typespb.JWTClaims_UserClaims{
			UserClaims: &typespb.UserJWTClaims{
				UserID: GetUserID(token),
				Email:  GetEmail(token),
			},
		}
	case HasServiceClaims(token):
		p.CustomClaims = &typespb.JWTClaims_ServiceClaims{
			ServiceClaims: &typespb.ServiceJWTClaims{
				ServiceID: GetServiceID(token),
			},
		}
	case HasDeviceClaims(token):
		p.CustomClaims = &typespb.JWTClaims_DeviceClaims{
			DeviceClaims: &typespb.DeviceJWTClaims{
				DeviceID: GetDeviceID(token),
				FleetID:  GetFleetID(token),
			},
		}
	}

	return p, nil
}

// SignToken signs the token using the given signing key.
func SignToken(token jwt.Token, signingKey string) (string, error) {
	key, err := jwk.FromRaw([]byte(signingKey))
	if err != nil {
		return "", err
	}
	signed, err := jwt.Sign(token, jwt.WithKey(jwa.HS256, key))
	if err != nil {
		return "", err
	}
	return string(signed), nil
}

// ParseToken parses the claim and validates that it was signed given signing key,
// and has the expected audience.
func ParseToken(tokenString string, signingKey string, audience string) (jwt.Token, error) {
	key, err := jwk.FromRaw([]byte(signingKey))
	if err != nil {
		return nil, err
	}
	return jwt.Parse([]byte(tokenString),
		jwt.WithKey(jwa.HS256, key),
		jwt.WithAudience(audience),
		jwt.WithValidate(true),
	)
}

// SignJWTClaims signs the claim using the given signing key.
func SignJWTClaims(claims *typespb.JWTClaims, signingKey string) (string, error) {
	token, err := ProtoToToken(claims)
	if err != nil {
		return "", err
	}
	return SignToken(token, signingKey)
}

// GetScopes fetches the Scopes from the custom claims.
func GetScopes(t jwt.Token) []string {
	claims := t.PrivateClaims()
	scopes, ok := claims["Scopes"]
	if !ok {
		return []string{}
	}
	return strings.Split(scopes.(string), ",")
}

// GetUserID fetches the UserID from the custom claims.
func GetUserID(t jwt.Token) string {
	claims := t.PrivateClaims()
	userID, ok := claims["UserID"]
	if !ok {
		return ""
	}
	return userID.(string)
}

// GetEmail fetches the Email from the custom claims.
func GetEmail(t jwt.Token) string {
	claims := t.PrivateClaims()
	email, ok := claims["Email"]
	if !ok {
		return ""
	}
	return email.(string)
}

// GetServiceID fetches the ServiceID from the custom claims.
func GetServiceID(t jwt.Token) string {
	claims := t.PrivateClaims()
	serviceID, ok := claims["ServiceID"]
	if !ok {
		return ""
	}
	return serviceID.(string)
}

// GetDeviceID fetches the DeviceID from the custom claims.
func GetDeviceID(t jwt.Token) string {
	claims := t.PrivateClaims()
	deviceID, ok := claims["DeviceID"]
	if !ok {
		return ""
	}
	return deviceID.(string)
}

// GetFleetID fetches the FleetID from the custom claims.
func GetFleetID(t jwt.Token) string {
	claims := t.PrivateClaims()
	fleetID, ok := claims["FleetID"]
	if !ok {
		return ""
	}
	return fleetID.(string)
}

// HasUserClaims checks if the custom claims include UserClaims.
func HasUserClaims(t jwt.Token) bool {
	claims := t.PrivateClaims()
	_, ok := claims["UserID"]
	return ok
}

// HasServiceClaims checks if the custom claims include ServiceClaims.
func HasServiceClaims(t jwt.Token) bool {
	claims := t.PrivateClaims()
	_, ok := claims["ServiceID"]
	return ok
}

// HasDeviceClaims checks if the custom claims include DeviceClaims.
func HasDeviceClaims(t jwt.Token) bool {
	claims := t.PrivateClaims()
	_, ok := claims["DeviceID"]
	return ok
}
