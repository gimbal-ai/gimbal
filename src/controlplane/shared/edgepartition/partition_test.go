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

package edgepartition_test

import (
	"fmt"
	"testing"

	"github.com/gofrs/uuid/v5"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"

	"gimletlabs.ai/gimlet/src/controlplane/shared/edgepartition"
)

func TestIntIDToHex(t *testing.T) {
	tests := []struct {
		input    int
		expected string
	}{
		{0, "000"},
		{10, "00a"},
		{255, "0ff"},
		{4095, "fff"},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("IntIDToHex(%d)", test.input), func(t *testing.T) {
			actual := edgepartition.IntIDToHex(test.input)
			assert.Equal(t, test.expected, actual)
		})
	}
}

func TestGenerateRange(t *testing.T) {
	tests := []struct {
		pCount   int
		pID      int
		expected []string
	}{
		{
			pCount:   256,
			pID:      0,
			expected: []string{"000", "001", "002", "003", "004", "005", "006", "007", "008", "009", "00a", "00b", "00c", "00d", "00e", "00f"},
		},
		{
			pCount:   256,
			pID:      1,
			expected: []string{"010", "011", "012", "013", "014", "015", "016", "017", "018", "019", "01a", "01b", "01c", "01d", "01e", "01f"},
		},
		{
			pCount:   256,
			pID:      255,
			expected: []string{"ff0", "ff1", "ff2", "ff3", "ff4", "ff5", "ff6", "ff7", "ff8", "ff9", "ffa", "ffb", "ffc", "ffd", "ffe", "fff"},
		},
	}

	for _, test := range tests {
		t.Run(fmt.Sprintf("TestGenerateRange_%d_%d", test.pCount, test.pID), func(t *testing.T) {
			viper.Set("partition_id", test.pID)
			viper.Set("partition_count", test.pCount)

			actual := edgepartition.GenerateRange()
			assert.Equal(t, test.expected, actual)
		})
	}
}

func TestEdgeIDToPartition(t *testing.T) {
	id, _ := uuid.FromString("6ba7b810-9dad-11d1-80b4-00c04fd430c8") // Example UUID
	expected := "6ba"

	actual := edgepartition.EdgeIDToPartition(id)
	assert.Equal(t, expected, actual)
}
