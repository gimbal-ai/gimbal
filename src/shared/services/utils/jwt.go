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
	"encoding/json"
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
		authorizations, _ := json.Marshal(m.UserClaims.Authorizations)
		builder.
			Claim("UserID", m.UserClaims.UserID).
			Claim("Email", m.UserClaims.Email).
			Claim("Authorizations", string(authorizations))
	case *typespb.JWTClaims_ServiceClaims:
		builder.Claim("ServiceID", m.ServiceClaims.ServiceID)
	case *typespb.JWTClaims_DeviceClaims:
		builder.Claim("DeviceID", m.DeviceClaims.DeviceID)
		builder.Claim("FleetID", m.DeviceClaims.FleetID)
		builder.Claim("DeployKeyID", m.DeviceClaims.DeployKeyID)
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
	scopes := []string{}
	claims := token.PrivateClaims()
	cScopes, ok := claims["Scopes"]
	if ok {
		scopes = strings.Split(cScopes.(string), ",")
	}
	p.Scopes = scopes

	switch {
	case HasUserClaims(token):
		p.CustomClaims = userTokenToProto(token)
	case HasServiceClaims(token):
		p.CustomClaims = serviceTokenToProto(token)
	case HasDeviceClaims(token):
		p.CustomClaims = deviceTokenToProto(token)
	}

	return p, nil
}

func userTokenToProto(token jwt.Token) *typespb.JWTClaims_UserClaims {
	claims := token.PrivateClaims()

	userID := ""
	cUserID, ok := claims["UserID"]
	if ok {
		userID, _ = cUserID.(string)
	}

	email := ""
	cEmail, ok := claims["Email"]
	if ok {
		email, _ = cEmail.(string)
	}

	authorizations := []*typespb.UserJWTClaims_AuthorizationDetails{}
	cAuth, ok := claims["Authorizations"]
	if ok {
		castedAuth, _ := cAuth.(string)
		err := json.Unmarshal([]byte(castedAuth), &authorizations)
		if err != nil {
			log.WithError(err).Error("Failed to unmarshal authorizations")
		}
	}

	return &typespb.JWTClaims_UserClaims{
		UserClaims: &typespb.UserJWTClaims{
			UserID:         userID,
			Email:          email,
			Authorizations: authorizations,
		},
	}
}

func serviceTokenToProto(token jwt.Token) *typespb.JWTClaims_ServiceClaims {
	claims := token.PrivateClaims()

	serviceID := ""
	cServiceID, ok := claims["ServiceID"]
	if ok {
		serviceID, _ = cServiceID.(string)
	}

	return &typespb.JWTClaims_ServiceClaims{
		ServiceClaims: &typespb.ServiceJWTClaims{
			ServiceID: serviceID,
		},
	}
}

func deviceTokenToProto(token jwt.Token) *typespb.JWTClaims_DeviceClaims {
	claims := token.PrivateClaims()

	deviceID := ""
	cDeviceID, ok := claims["DeviceID"]
	if ok {
		deviceID, _ = cDeviceID.(string)
	}

	fleetID := ""
	cFleetID, ok := claims["FleetID"]
	if ok {
		fleetID, _ = cFleetID.(string)
	}

	deployKeyID := ""
	cDeployKeyID, ok := claims["DeployKeyID"]
	if ok {
		deployKeyID, _ = cDeployKeyID.(string)
	}

	return &typespb.JWTClaims_DeviceClaims{
		DeviceClaims: &typespb.DeviceJWTClaims{
			DeviceID:    deviceID,
			FleetID:     fleetID,
			DeployKeyID: deployKeyID,
		},
	}
}

// SignToken signs the token using the given signing key.
func SignToken(token jwt.Token, signingKey string) (string, error) {
	key, err := jwk.ParseKey([]byte(signingKey))
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
	key, err := jwk.ParseKey([]byte(signingKey))
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
