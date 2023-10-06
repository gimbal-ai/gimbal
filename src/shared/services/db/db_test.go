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

package db_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	idDb "gimletlabs.ai/gimlet/src/shared/services/db"
	"gimletlabs.ai/gimlet/src/shared/services/pgtest"
)

func TestCreateIdempotentTx(t *testing.T) {
	db, teardown, err := pgtest.SetupTestDB(nil)
	require.NoError(t, err)
	require.NotNil(t, db)
	require.NotNil(t, teardown)
	defer teardown()

	// Add idempotency table. In general, we expect this table to be created by the service creator in the migrations.
	_, err = db.Exec("CREATE TABLE IF NOT EXISTS test_idempotency_keys (idempotency_key text UNIQUE, created_at timestamp NOT NULL DEFAULT NOW())")
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO test_idempotency_keys (idempotency_key) VALUES ('test-key')")
	require.NoError(t, err)

	// Create fake test table.
	_, err = db.Exec("CREATE TABLE IF NOT EXISTS test (test_col text)")
	require.NoError(t, err)

	tx1, err := idDb.CreateIdempotentTx(db, "test-key", "test")
	require.ErrorContains(t, err, "not idempotent")
	require.NoError(t, tx1.Rollback())

	tx2, err := idDb.CreateIdempotentTx(db, "new-key", "test")
	require.NoError(t, err)

	_, err = tx2.Exec("INSERT INTO test (test_col) VALUES ('hello')")
	require.NoError(t, err)
	require.NoError(t, tx2.Commit())

	// Verify db has been updated.
	rows1, err := db.Queryx("SELECT * FROM test")
	require.NoError(t, err)
	defer rows1.Close()
	assert.True(t, rows1.Next())
	testKey := ""
	err = rows1.Scan(&testKey)
	require.NoError(t, err)
	assert.Equal(t, "hello", testKey)

	rows2, err := db.Queryx("SELECT count(*) FROM test_idempotency_keys")
	assert.NoError(t, err)
	defer rows2.Close()
	require.True(t, rows2.Next())
	count := 0
	err = rows2.Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 2, count)
}
