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

package pg

import (
	"context"
	"embed"
	"fmt"

	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	"github.com/golang-migrate/migrate/v4/source/iofs"
	"github.com/jmoiron/sqlx"
)

// PerformMigrationsWithEmbed uses the passed in embed to perform postgres DB migrations.
func PerformMigrationsWithEmbed(db *sqlx.DB, migrationTable string, assetSource *embed.FS, sourceDir string) error {
	ctx := context.Background()

	conn, err := db.DB.Conn(ctx)
	if err != nil {
		return err
	}
	defer conn.Close()

	driver, err := postgres.WithConnection(ctx, conn, &postgres.Config{
		MigrationsTable: migrationTable,
	})
	if err != nil {
		return fmt.Errorf("failed to load migrations: %w", err)
	}

	d, err := iofs.New(assetSource, sourceDir)
	if err != nil {
		return fmt.Errorf("failed to load schema: %w", err)
	}
	defer d.Close()

	mg, err := migrate.NewWithInstance(
		"iofs", d, "postgres", driver)
	if err != nil {
		return fmt.Errorf("failed to load migrations: %w", err)
	}

	if err = mg.Up(); err != nil && err != migrate.ErrNoChange {
		return fmt.Errorf("migrations failed: %w", err)
	}

	return nil
}
