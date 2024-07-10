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
	"io"
	"testing"

	"github.com/gofrs/uuid/v5"
	log "github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
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
	err := testDB.Reset()
	require.NoError(t, err)
	db := testDB.DB()

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
	err := testDB.Reset()
	require.NoError(b, err)
	db := testDB.DB()

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
	err := testDB.Reset()
	require.NoError(b, err)
	db := testDB.DB()
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
	err := testDB.Reset()
	require.NoError(b, err)
	db := testDB.DB()
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
	err := testDB.Reset()
	require.NoError(b, err)
	db := testDB.DB()
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
