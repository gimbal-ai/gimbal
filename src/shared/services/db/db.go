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

package db

import (
	"errors"
	"fmt"

	// This is required to get the "pgx" driver.
	_ "github.com/jackc/pgx/v5/stdlib"
	"github.com/jmoiron/sqlx"
)

var errIdempotencyTxFailed = errors.New("transaction not idempotent")

// CreateIdempotentTx creates a transaction for a query that will only run if the idempotency key does not yet exist.
func CreateIdempotentTx(db *sqlx.DB, key, svc string) (*sqlx.Tx, error) {
	tx, err := db.Beginx()
	if err != nil {
		return tx, errIdempotencyTxFailed
	}
	// Will fail if unique key already exists.
	_, err = tx.Exec(fmt.Sprintf("INSERT INTO %s_idempotency_keys (idempotency_key) VALUES ('%s')", svc, key))
	if err != nil {
		return tx, errIdempotencyTxFailed
	}

	return tx, nil
}
