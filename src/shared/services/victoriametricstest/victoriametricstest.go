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

package victoriametricstest

import (
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/ory/dockertest/v3"
	"github.com/ory/dockertest/v3/docker"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/shared/services/victoriametrics"
	"gimletlabs.ai/gimlet/src/testutils/dockertestutils"
)

var (
	ErrEmptyResult           = errors.New("no such metric available")
	ErrUnsupportedMetricType = errors.New("metric type is unsupported")
)

func IsMetricAvailable(ctx context.Context, conn v1.API, query string) error {
	res, _, err := conn.QueryRange(ctx, query, v1.Range{
		Start: time.Now().Add(-1 * time.Hour),
		End:   time.Now(),
		Step:  60 * time.Second,
	})
	if err != nil {
		return err
	}
	var l int
	switch res.Type() {
	case model.ValVector:
		items := res.(model.Vector)
		l = items.Len()
	case model.ValMatrix:
		items := res.(model.Matrix)
		l = items.Len()
	default:
		return fmt.Errorf("%w: %s", ErrUnsupportedMetricType, res.Type().String())
	}
	if l < 1 {
		return ErrEmptyResult
	}
	return nil
}

func WaitForMetrics(t *testing.T, conn v1.API, q string) {
	bo := backoff.NewExponentialBackOff()
	bo.MaxElapsedTime = 30 * time.Second
	bo.MaxInterval = 1 * time.Second
	bo.InitialInterval = 1 * time.Second

	// Victoriametric seems to be caching the result and takes a while to invalidate the cache.
	// So sleeping here actually speeds up the test by increasing the chance that the metric
	// is available the first time you query.
	time.Sleep(3 * time.Second)
	err := backoff.Retry(func() error {
		err := IsMetricAvailable(context.Background(), conn, q)
		if err != nil && !errors.Is(err, ErrEmptyResult) {
			t.Fatalf("failed to wait for metrics: %v", err)
		}
		return err
	}, bo)

	require.Nil(t, err)
}

// SetupTestVictoriaMetrics sets up a test instance for victoriametrics.
func SetupTestVictoriaMetrics() (v1.API, func(), error) {
	pool, err := dockertest.NewPool("")
	if err != nil {
		return nil, nil, fmt.Errorf("connect to docker failed: %w", err)
	}

	imageRepo := "victoriametrics/victoria-metrics"
	imageTag := "v1.93.6"
	err = dockertestutils.LoadOrFetchImage(pool, imageRepo, imageTag)
	if err != nil {
		return nil, nil, err
	}

	resource, err := pool.RunWithOptions(
		&dockertest.RunOptions{
			Repository: imageRepo,
			Tag:        imageTag,
			Cmd:        []string{"-search.latencyOffset=0s", "-search.disableCache"},
		}, func(config *docker.HostConfig) {
			config.AutoRemove = true
			config.RestartPolicy = docker.RestartPolicy{Name: "no"}
			config.Mounts = []docker.HostMount{
				{
					Target: "/victoria-metrics-data",
					Type:   "tmpfs",
					TempfsOptions: &docker.TempfsOptions{
						SizeBytes: 100 * 1024 * 1024,
					},
				},
			}
			config.CPUCount = 1
			config.Memory = 512 * 1024 * 1024
			config.MemorySwap = 0
			config.MemorySwappiness = 0
		},
	)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to run docker pool: %w", err)
	}
	// Set a 5 minute expiration on resources.
	err = resource.Expire(300)
	if err != nil {
		return nil, nil, err
	}

	// Single node mode uses the same host/port for inserts as well as selects.
	viper.Set("victoriametrics_insert_scheme", "http")
	viper.Set("victoriametrics_insert_host", resource.Container.NetworkSettings.Gateway)
	viper.Set("victoriametrics_insert_port", resource.GetPort("8428/tcp"))

	viper.Set("victoriametrics_select_scheme", "http")
	viper.Set("victoriametrics_select_host", resource.Container.NetworkSettings.Gateway)
	viper.Set("victoriametrics_select_port", resource.GetPort("8428/tcp"))

	var conn v1.API
	if err = pool.Retry(func() error {
		log.SetLevel(log.WarnLevel)
		log.Info("trying to connect")
		conn = victoriametrics.MustConnectVictoriaMetricsSelect()
		_, _, err := conn.Query(context.Background(), "up", time.Now(), v1.WithTimeout(5*time.Second))
		return err
	}); err != nil {
		return nil, nil, fmt.Errorf("failed to create victoriametrics on docker: %w", err)
	}
	log.SetLevel(log.InfoLevel)

	return conn, func() {
		if err := pool.Purge(resource); err != nil {
			log.WithError(err).Error("could not purge docker resource")
		}
	}, nil
}
