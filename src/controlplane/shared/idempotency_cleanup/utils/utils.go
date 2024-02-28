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

package utils

import (
	"context"
	"fmt"
	"time"

	"github.com/jmoiron/sqlx"
	log "github.com/sirupsen/logrus"
)

var tablesQuery = `select
table_name
from information_schema.tables
where table_name like '%_idempotency_key'
and table_schema not in ('information_schema', 'pg_catalog')
and table_type = 'BASE TABLE'
order by table_name,
	table_schema`

// ExpireKeys will expire any idempotency keys greater than the expiry time.
func ExpireKeys(ctx context.Context, db *sqlx.DB, expiry time.Duration) error {
	// Find all idempotency tables.
	rows, err := db.QueryxContext(ctx, tablesQuery)
	if err != nil {
		return err
	}
	defer rows.Close()
	var tables []string

	for rows.Next() {
		var tableName string
		err = rows.Scan(&tableName)
		if err != nil {
			return err
		}
		tables = append(tables, tableName)
	}

	// Clean up keys for each table.
	for _, table := range tables {
		expireTime := time.Now().Add(-1 * expiry)
		_, err := db.ExecContext(ctx, fmt.Sprintf("DELETE FROM %s WHERE created_at < $1", table), expireTime)
		if err != nil {
			log.WithError(err).Error(fmt.Sprintf("Failed to clean up %s... Continuing.", table))
		}
	}
	return nil
}
