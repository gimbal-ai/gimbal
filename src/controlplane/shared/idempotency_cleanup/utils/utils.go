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
