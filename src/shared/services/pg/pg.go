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

package pg

import (
	"fmt"
	"net"
	"net/url"
	"time"

	// This is required to get the "pgx" driver.
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/collectors"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

const (
	retryAttempts = 5
	retryDelay    = 1 * time.Second
)

func init() {
	pflag.Uint32("postgres_port", 5432, "The port for postgres database")
	pflag.String("postgres_hostname", "localhost", "The hostname for postgres database")
	pflag.String("postgres_db", "test", "The name of the database to use")
	pflag.String("postgres_username", "gml", "The username in the postgres database")
	pflag.String("postgres_password", "gml", "The password in the postgres database")
	pflag.Bool("postgres_ssl", false, "Enable ssl for postgres")
}

// postgresConfig contains all the necessary configurations to connect to a Postgres DB.
type postgresConfig struct {
	port       int32
	hostname   string
	db         string
	username   string
	password   string
	sslEnabled bool
}

func dbURI(config *postgresConfig) string {
	sslMode := "require"
	if !config.sslEnabled {
		sslMode = "disable"
	}

	v := url.Values{}
	v.Set("sslmode", sslMode)

	u := url.URL{
		Scheme:   "postgres",
		Host:     net.JoinHostPort(config.hostname, fmt.Sprintf("%d", config.port)),
		User:     url.UserPassword(config.username, config.password),
		Path:     config.db,
		RawQuery: v.Encode(),
	}

	return u.String()
}

func mustCreatePostgresDB(config *postgresConfig) *sqlx.DB {
	dbURI := dbURI(config)
	log.WithField("db_hostname", config.hostname).
		WithField("db_port", config.port).
		WithField("db_name", config.db).
		WithField("db_username", config.username).
		Info("Setting up database")

	db, err := sqlx.Open("pgx", dbURI)
	if err != nil {
		log.WithError(err).Fatalf("failed to setup database connection")
	}

	db.SetMaxIdleConns(5)
	db.SetConnMaxLifetime(30 * time.Minute)
	db.SetMaxOpenConns(10)

	// It's possible we already registered a prometheus collector with multiple DB connections.
	_ = prometheus.Register(
		collectors.NewDBStatsCollector(db.DB, config.db))
	return db
}

func mustConnectPostgresDB(config *postgresConfig) *sqlx.DB {
	db := mustCreatePostgresDB(config)
	var err error
	for i := retryAttempts; i >= 0; i-- {
		err = db.Ping()
		if err == nil {
			log.Info("Connected to Postgres")
			break
		}
		if i > 0 {
			log.WithError(err).Error("failed to connect to DB, retrying")
			time.Sleep(retryDelay)
		}
	}

	if err != nil {
		log.WithError(err).Fatalf("failed to initialized database connection")
	}
	return db
}

func defaultDBConfig() *postgresConfig {
	return &postgresConfig{
		port:       viper.GetInt32("postgres_port"),
		hostname:   viper.GetString("postgres_hostname"),
		db:         viper.GetString("postgres_db"),
		username:   viper.GetString("postgres_username"),
		password:   viper.GetString("postgres_password"),
		sslEnabled: viper.GetBool("postgres_ssl"),
	}
}

// DefaultDBURI returns the URI string for the default postgres instance based on flags/env vars.
func DefaultDBURI() string {
	return dbURI(defaultDBConfig())
}

// MustCreateDefaultPostgresDB creates a postgres DB instance.
func MustCreateDefaultPostgresDB() *sqlx.DB {
	return mustCreatePostgresDB(defaultDBConfig())
}

// MustConnectDefaultPostgresDB tries to connect to default postgres database as defined by the environment
// variables/flags.
func MustConnectDefaultPostgresDB() *sqlx.DB {
	return mustConnectPostgresDB(defaultDBConfig())
}
