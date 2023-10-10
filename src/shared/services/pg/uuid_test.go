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

package pg_test

import (
	"io"
	"testing"

	"github.com/gofrs/uuid/v5"
	log "github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/shared/services/pgtest"
)

const (
	createTableStmt = "CREATE TABLE IF NOT EXISTS test (id UUID DEFAULT gen_random_uuid(), name TEXT)"

	insertStmt = "INSERT INTO test (id, name) VALUES ($1, $2)"
	queryStmt  = "SELECT id FROM test WHERE name = $1"
)

func init() {
	// The pgsetup makes it difficult to parse benchmark results.
	// So we silence the log output from SetupTestDB (and any other log calls).
	log.SetOutput(io.Discard)
}

func TestUUIDStrings(t *testing.T) {
	db, teardown, err := pgtest.SetupTestDB(nil)
	require.NoError(t, err)
	require.NotNil(t, db)
	require.NotNil(t, teardown)
	defer teardown()

	// Create fake test table.
	_, err = db.Exec(createTableStmt)
	require.NoError(t, err)

	uuid1 := uuid.Must(uuid.NewV4())
	uuid2 := uuid.Must(uuid.NewV4())

	// Insert as uuid.UUID
	_, err = db.Exec(insertStmt, uuid1, "uuid1")
	require.NoError(t, err)

	// Query as string
	row := db.QueryRowx(queryStmt, "uuid1")
	var uuidStringScanned string
	err = row.Scan(&uuidStringScanned)
	require.NoError(t, err)
	assert.Equal(t, uuidStringScanned, uuid1.String())

	// Insert as string
	_, err = db.Exec(insertStmt, uuid2.String(), "uuid2")
	require.NoError(t, err)

	// Query as uuid.UUID
	row = db.QueryRowx(queryStmt, "uuid2")
	var uuidScanned uuid.UUID
	err = row.Scan(&uuidScanned)
	require.NoError(t, err)
	assert.Equal(t, uuidScanned, uuid2)
}

func BenchmarkInserts_UUID(b *testing.B) {
	db, teardown, err := pgtest.SetupTestDB(nil)
	require.NoError(b, err)
	require.NotNil(b, db)
	require.NotNil(b, teardown)
	defer teardown()
	// Stop timer before we start teardown.
	defer b.StopTimer()

	_, err = db.Exec(createTableStmt)
	require.NoError(b, err)
	uuid1 := uuid.Must(uuid.NewV4())

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = db.Exec(insertStmt, uuid1, "uuid1")
	}
}

func BenchmarkInserts_String(b *testing.B) {
	db, teardown, err := pgtest.SetupTestDB(nil)
	require.NoError(b, err)
	require.NotNil(b, db)
	require.NotNil(b, teardown)
	defer teardown()
	// Stop timer before we start teardown.
	defer b.StopTimer()

	_, err = db.Exec(createTableStmt)
	require.NoError(b, err)
	uuid1 := uuid.Must(uuid.NewV4()).String()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = db.Exec(insertStmt, uuid1, "uuid1")
	}
}

func BenchmarkReads_UUID(b *testing.B) {
	db, teardown, err := pgtest.SetupTestDB(nil)
	require.NoError(b, err)
	require.NotNil(b, db)
	require.NotNil(b, teardown)
	defer teardown()
	// Stop timer before we start teardown.
	defer b.StopTimer()

	_, err = db.Exec(createTableStmt)
	require.NoError(b, err)
	uuid1 := uuid.Must(uuid.NewV4())
	_, err = db.Exec(insertStmt, uuid1, "uuid1")
	require.NoError(b, err)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		row := db.QueryRowx(queryStmt, "uuid1")
		var scan uuid.UUID
		_ = row.Scan(&scan)
	}
}

func BenchmarkReads_String(b *testing.B) {
	db, teardown, err := pgtest.SetupTestDB(nil)
	require.NoError(b, err)
	require.NotNil(b, db)
	require.NotNil(b, teardown)
	defer teardown()
	// Stop timer before we start teardown.
	defer b.StopTimer()

	_, err = db.Exec(createTableStmt)
	require.NoError(b, err)
	uuid1 := uuid.Must(uuid.NewV4())
	_, err = db.Exec(insertStmt, uuid1, "uuid1")
	require.NoError(b, err)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		row := db.QueryRowx(queryStmt, "uuid1")
		var scan string
		_ = row.Scan(&scan)
	}
}
