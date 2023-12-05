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

package utils_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/controlplane/shared/idempotency_cleanup/utils"
	"gimletlabs.ai/gimlet/src/shared/services/pgtest"
)

func TestExpireKeys(t *testing.T) {
	db, err := pgtest.SetupTestDB(t, nil)
	require.NoError(t, err)
	require.NotNil(t, db)

	now := time.Now()
	// 5 days ago.
	expired1 := now.Add(-1 * 5 * 24 * time.Hour)
	// 2 days ago.
	expired2 := now.Add(-1 * 2 * 24 * time.Hour)
	// 1 day from now.
	future1 := now.Add(1 * 24 * time.Hour)
	// 2 days from now.
	future2 := now.Add(2 * 24 * time.Hour)

	_, err = db.Exec("CREATE TABLE IF NOT EXISTS test1_idempotency_keys (idempotency_key text UNIQUE, created_at timestamp NOT NULL DEFAULT NOW())")
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO test1_idempotency_keys (idempotency_key, created_at) VALUES ('test-key', $1)", future1)
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO test1_idempotency_keys (idempotency_key, created_at) VALUES ('expired-key', $1)", expired1)
	require.NoError(t, err)

	_, err = db.Exec("CREATE TABLE IF NOT EXISTS test2_idempotency_keys (idempotency_key text UNIQUE, created_at timestamp NOT NULL DEFAULT NOW())")
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO test2_idempotency_keys (idempotency_key, created_at) VALUES ('test-key', $1)", future2)
	require.NoError(t, err)
	_, err = db.Exec("INSERT INTO test2_idempotency_keys (idempotency_key, created_at) VALUES ('expired-key', $1)", expired2)
	require.NoError(t, err)

	err = utils.ExpireKeys(db, 24*time.Hour)
	require.NoError(t, err)

	rows, err := db.Queryx("SELECT idempotency_key FROM test1_idempotency_keys")
	require.NoError(t, err)
	defer rows.Close()
	var keys []string
	for rows.Next() {
		var k string
		err = rows.Scan(&k)
		require.NoError(t, err)
		keys = append(keys, k)
	}
	assert.ElementsMatch(t, keys, []string{"test-key"})

	rows2, err := db.Queryx("SELECT idempotency_key FROM test2_idempotency_keys")
	require.NoError(t, err)
	defer rows2.Close()
	var keys2 []string
	for rows2.Next() {
		var k string
		err = rows2.Scan(&k)
		require.NoError(t, err)
		keys2 = append(keys2, k)
	}
	assert.ElementsMatch(t, keys2, []string{"test-key"})
}
