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

package victoriametricstest_test

import (
	"testing"

	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/shared/services/victoriametricstest"
)

func TestSetupTestVictoriaMetrics(t *testing.T) {
	conn, teardown, err := victoriametricstest.SetupTestVictoriaMetrics()

	require.NoError(t, err)
	require.NotNil(t, conn)
	require.NotNil(t, teardown)

	teardown()
}
