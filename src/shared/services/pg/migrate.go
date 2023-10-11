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

package pg

import (
	"embed"
	"fmt"

	"github.com/golang-migrate/migrate/v4"
	"github.com/golang-migrate/migrate/v4/database/postgres"
	"github.com/golang-migrate/migrate/v4/source/iofs"
	"github.com/jmoiron/sqlx"
)

// PerformMigrationsWithEmbed uses the passed in embed to perform postgres DB migrations.
func PerformMigrationsWithEmbed(db *sqlx.DB, migrationTable string, assetSource *embed.FS, sourceDir string) error {
	driver, err := postgres.WithInstance(db.DB, &postgres.Config{
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
