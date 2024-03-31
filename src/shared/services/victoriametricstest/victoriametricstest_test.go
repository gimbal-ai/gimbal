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
	"context"
	"testing"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/prometheus/common/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/shared/services/victoriametrics"
	"gimletlabs.ai/gimlet/src/shared/services/victoriametricstest"
)

func TestSetupTestVictoriaMetrics(t *testing.T) {
	conn, err := victoriametricstest.SetupTestVictoriaMetrics(t)

	require.NoError(t, err)
	assert.NotNil(t, conn)
}

func TestSetupTestVictoriaMetrics_tSimpleImportExport(t *testing.T) {
	conn, err := victoriametricstest.SetupTestVictoriaMetrics(t)

	require.NoError(t, err)
	require.NotNil(t, conn)

	err = victoriametrics.InsertPrometheusMetrics(`foo{bar="baz"} 123`)
	require.NoError(t, err)

	query := `foo{bar="baz"}`

	bo := backoff.NewExponentialBackOff()
	bo.MaxElapsedTime = 30 * time.Second
	bo.MaxInterval = 1 * time.Second
	bo.InitialInterval = 1 * time.Second

	// Victoriametric seems to be caching the result and takes a while to invalidate the cache.
	// So sleeping here actually speeds up the test by increasing the chance that the metric
	// is available the first time you query.
	time.Sleep(3 * time.Second)
	err = backoff.Retry(func() error {
		return victoriametricstest.IsMetricAvailable(context.Background(), conn, query)
	}, bo)
	if err != nil {
		t.Fatalf("Could not fetch metric. backoff err: %s", err)
	}

	res, warn, err := conn.Query(context.Background(), query, time.Now())
	require.NoError(t, err)
	assert.Empty(t, warn)
	assert.Equal(t, model.ValVector, res.Type())
	items, ok := res.(model.Vector)
	require.True(t, ok)

	require.Equal(t, 1, items.Len())
	assert.True(t, items[0].Metric.Equal(model.Metric(model.LabelSet{"__name__": "foo", "bar": "baz"})))
	assert.True(t, items[0].Value.Equal(123))
}
