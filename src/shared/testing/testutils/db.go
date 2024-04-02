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

package testutils

import (
	"testing"

	"github.com/jmoiron/sqlx"
	"github.com/stretchr/testify/require"
)

func ValidateEnumValues(t *testing.T, db *sqlx.DB, sqlEnumName string, protoEnumName map[int32]string) {
	rows, err := db.Query("SELECT unnest(enum_range(NULL::" + sqlEnumName + "))")
	require.NoError(t, err)

	dbEnumValues := []string{}
	for rows.Next() {
		var enumValue string
		err = rows.Scan(&enumValue)
		require.NoError(t, err)
		dbEnumValues = append(dbEnumValues, enumValue)
	}

	values := []string{}
	for _, v := range protoEnumName {
		values = append(values, v)
	}

	require.ElementsMatch(t, values, dbEnumValues)
}
