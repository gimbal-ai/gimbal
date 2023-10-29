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
	// This is required to get the "pgx" driver.
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

func init() {
	pflag.Uint32("tsdb_port", 5432, "The port for TSDB")
	pflag.String("tsdb_hostname", "gml-timescaledb", "The hostname for TSDB")
	pflag.String("tsdb_db", "tsdb", "The name of the database to use in TSDB")
	pflag.String("tsdb_username", "gml", "The username in TSDB")
	pflag.String("tsdb_password", "gml", "The password in TSDB")
	pflag.Bool("tsdb_ssl", true, "Enable ssl for TSDB")
}

func defaultTSDBConfig() *postgresConfig {
	return &postgresConfig{
		port:       viper.GetInt32("tsdb_port"),
		hostname:   viper.GetString("tsdb_hostname"),
		db:         viper.GetString("tsdb_db"),
		username:   viper.GetString("tsdb_username"),
		password:   viper.GetString("tsdb_password"),
		sslEnabled: viper.GetBool("tsdb_ssl"),
	}
}

// DefaultDBURI returns the URI string for the default timescaledb instance based on flags/env vars.
func DefaultTSDBURI() string {
	return dbURI(defaultTSDBConfig())
}

// MustCreateDefaultTSDB creates a timescale DB instance.
func MustCreateDefaultTSDB() *sqlx.DB {
	return mustCreatePostgresDB(defaultTSDBConfig())
}

// MustConnectDefaultTSDB tries to connect to default timescaledb as defined by the environment
// variables/flags.
func MustConnectDefaultTSDB() *sqlx.DB {
	return mustConnectPostgresDB(defaultTSDBConfig())
}
