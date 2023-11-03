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
	"fmt"
	"os"
	"time"

	"github.com/bazelbuild/rules_go/go/runfiles"
	"github.com/ory/dockertest/v3"
	"github.com/ory/dockertest/v3/docker"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	"github.com/prometheus/common/model"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/viper"

	"gimletlabs.ai/gimlet/src/shared/services/victoriametrics"
)

func IsMetricAvailable(ctx context.Context, conn v1.API, query string) error {
	res, _, err := conn.Query(ctx, query, time.Now())
	if err != nil {
		return err
	}
	switch res.Type() {
	case model.ValVector:
		items := res.(model.Vector)
		if items.Len() < 1 {
			return fmt.Errorf("empty results")
		}
		return nil
	}
	return fmt.Errorf("metric type %s not yet supported", res.Type().String())
}

// SetupTestVictoriaMetrics sets up a test instance for victoriametrics.
func SetupTestVictoriaMetrics() (v1.API, func(), error) {
	pool, err := dockertest.NewPool("")
	if err != nil {
		return nil, nil, fmt.Errorf("connect to docker failed: %w", err)
	}

	// Load image.
	imgPath, err := runfiles.Rlocation("gml/src/shared/services/victoriametricstest/vm_image/tarball.tar")
	if err != nil {
		return nil, nil, fmt.Errorf("failed to find image runfile")
	}
	f, err := os.Open(imgPath)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to read tarball")
	}
	err = pool.Client.LoadImage(docker.LoadImageOptions{
		InputStream: f,
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to load image")
	}

	resource, err := pool.RunWithOptions(
		&dockertest.RunOptions{
			Repository: "victoriametrics/victoria-metrics",
			Tag:        "v1.93.6",
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
		return nil, nil, fmt.Errorf("Failed to run docker pool: %w", err)
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
		return nil, nil, fmt.Errorf("failed to create postgres on docker: %w", err)
	}
	log.SetLevel(log.InfoLevel)

	return conn, func() {
		if err := pool.Purge(resource); err != nil {
			log.WithError(err).Error("could not purge docker resource")
		}
	}, nil
}
