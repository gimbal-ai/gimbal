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

package env_test

import (
	"testing"

	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"

	"gimletlabs.ai/gimlet/src/shared/services/env"
)

func TestEnv_New(t *testing.T) {
	viper.Set("jwt_signing_key", "the-jwt-key")
	viper.Set("pod_name", "thepod")

	env := env.New("svc")
	assert.Equal(t, "the-jwt-key", env.JWTSigningKey())
	assert.Equal(t, "svc", env.ServiceName())
	assert.Equal(t, "thepod", env.PodName())
}