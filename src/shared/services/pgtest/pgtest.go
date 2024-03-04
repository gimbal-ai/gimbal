/*
 * Copyright 2018- The Pixie Authors.
 * Modifications Copyright 2023- Gimlet Labs, Inc.
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

package pgtest

import (
	"embed"
	"fmt"
	"testing"

	"github.com/jmoiron/sqlx"
	"github.com/ory/dockertest/v3"
	"github.com/ory/dockertest/v3/docker"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/viper"

	"gimletlabs.ai/gimlet/src/shared/services/pg"
	"gimletlabs.ai/gimlet/src/testutils/dockertestutils"
)

type testDB struct {
	db                    *sqlx.DB
	schemaSourceDirectory string
}

// TestDBOpt is an option to the testing DB.
type TestDBOpt func(*testDB)

// WithSchemaDirectory allows configuration of the schema directory.
func WithSchemaDirectory(dir string) TestDBOpt {
	return func(d *testDB) {
		d.schemaSourceDirectory = dir
	}
}

// SetupTestDB sets up a test database instance and applies migrations.
func SetupTestDB(t testing.TB, schemaSource *embed.FS, opts ...TestDBOpt) (*sqlx.DB, error) {
	d := &testDB{
		schemaSourceDirectory: ".",
	}
	for _, opt := range opts {
		opt(d)
	}

	pool, err := dockertest.NewPool("")
	if err != nil {
		return nil, fmt.Errorf("connect to docker failed: %w", err)
	}

	imageRepo := "pgvector/pgvector"
	imageTag := "pg15"
	err = dockertestutils.LoadOrFetchImage(pool, imageRepo, imageTag)
	if err != nil {
		return nil, err
	}

	const dbName = "testdb"
	resource, err := pool.RunWithOptions(
		&dockertest.RunOptions{
			Repository: imageRepo,
			Tag:        imageTag,
			Env:        []string{"POSTGRES_PASSWORD=secret", "POSTGRES_DB=" + dbName},
		}, func(config *docker.HostConfig) {
			config.AutoRemove = true
			config.RestartPolicy = docker.RestartPolicy{Name: "no"}
			config.Mounts = []docker.HostMount{
				{
					Target: "/var/lib/postgresql/data",
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
		return nil, fmt.Errorf("failed to run docker pool: %w", err)
	}
	// Set a 5 minute expiration on resources.
	err = resource.Expire(300)
	if err != nil {
		return nil, err
	}

	viper.Set("postgres_port", resource.GetPort("5432/tcp"))
	viper.Set("postgres_hostname", resource.Container.NetworkSettings.Gateway)
	viper.Set("postgres_db", dbName)
	viper.Set("postgres_username", "postgres")
	viper.Set("postgres_password", "secret")

	// Intentionally restrict the number of DB conns. This is to ensure that
	// there are no funcs that consume more than one conn simultaneously since
	// that causes deadlocks.
	viper.Set("postgres_max_idle_conns", 1)
	viper.Set("postgres_max_open_conns", 1)

	if err = pool.Retry(func() error {
		log.SetLevel(log.WarnLevel)
		log.Info("trying to connect")
		d.db = pg.MustCreateDefaultPostgresDB()
		return d.db.Ping()
	}); err != nil {
		return nil, fmt.Errorf("failed to create postgres on docker: %w", err)
	}
	log.SetLevel(log.InfoLevel)

	if schemaSource != nil {
		err := pg.PerformMigrationsWithEmbed(d.db, "test_migrations", schemaSource, d.schemaSourceDirectory)
		if err != nil {
			return nil, fmt.Errorf("migrations failed: %w", err)
		}
	}

	t.Cleanup(func() {
		if d.db != nil {
			d.db.Close()
		}

		if err := pool.Purge(resource); err != nil {
			log.WithError(err).Error("could not purge docker resource")
		}
	})

	return d.db, nil
}
