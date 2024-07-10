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

package pg_test

import (
	"context"
	"database/sql"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/grpc/metadata"

	"gimletlabs.ai/gimlet/src/shared/services/pg"
)

func TestCreateIdempotentTx(t *testing.T) {
	err := testDB.Reset()
	require.NoError(t, err)
	db := testDB.DB()

	// Add idempotency table. In general, we expect this table to be created by the service creator in the migrations.
	_, err = db.Exec("CREATE TABLE IF NOT EXISTS test_idempotency_key (idempotency_key text UNIQUE, created_at timestamp NOT NULL DEFAULT NOW())")
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO test_idempotency_key (idempotency_key) VALUES ('test-key')")
	require.NoError(t, err)

	// Create fake test table.
	_, err = db.Exec("CREATE TABLE IF NOT EXISTS test (test_col text)")
	require.NoError(t, err)

	md1 := map[string][]string{
		"x-idempotency-key": {"test-key"},
	}
	ctx1 := metadata.NewIncomingContext(context.Background(), md1)
	tx1, err := pg.CreateIdempotentTx(ctx1, db, "test")
	require.ErrorIs(t, err, pg.ErrIdempotencyTxFailed)
	require.ErrorIs(t, tx1.Rollback(), sql.ErrTxDone)

	md2 := map[string][]string{
		"x-idempotency-key": {"new-key"},
	}
	ctx2 := metadata.NewIncomingContext(context.Background(), md2)
	tx2, err := pg.CreateIdempotentTx(ctx2, db, "test")
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

	rows1.Close()

	rows2, err := db.Queryx("SELECT count(*) FROM test_idempotency_key")
	require.NoError(t, err)
	defer rows2.Close()
	require.True(t, rows2.Next())
	count := 0
	err = rows2.Scan(&count)
	require.NoError(t, err)
	assert.Equal(t, 2, count)

	rows2.Close()

	tx3, err := pg.CreateIdempotentTx(context.Background(), db, "test")
	require.ErrorIs(t, err, pg.ErrIdempotencyKeyMissing)
	require.ErrorIs(t, tx3.Rollback(), sql.ErrTxDone)
}
