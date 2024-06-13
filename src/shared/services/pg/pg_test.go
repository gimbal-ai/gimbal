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

package pg_test

import (
	"os"
	"testing"

	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"

	"gimletlabs.ai/gimlet/src/shared/services/pg"
	"gimletlabs.ai/gimlet/src/shared/services/pgtest"
)

var testDB *pgtest.TestDB

func TestDefaultDBURI(t *testing.T) {
	currSettings := viper.AllSettings()
	t.Cleanup(func() {
		for k, v := range currSettings {
			viper.Set(k, v)
		}
	})

	viper.Set("postgres_port", 5000)
	viper.Set("postgres_hostname", "postgres-host")
	viper.Set("postgres_db", "thedb")
	viper.Set("postgres_username", "user")
	viper.Set("postgres_password", "pass")

	t.Run("With SSL", func(t *testing.T) {
		viper.Set("postgres_ssl", true)
		assert.Equal(t, "postgres://user:pass@postgres-host:5000/thedb?sslmode=require", pg.DefaultDBURI())
	})

	t.Run("Without SSL", func(t *testing.T) {
		viper.Set("postgres_ssl", false)
		assert.Equal(t, "postgres://user:pass@postgres-host:5000/thedb?sslmode=disable", pg.DefaultDBURI())
	})
}

func TestMain(m *testing.M) {
	db, dbCleanup := pgtest.MustSetupTestDB(nil)
	testDB = db

	exitVal := m.Run()
	dbCleanup()

	os.Exit(exitVal)
}
