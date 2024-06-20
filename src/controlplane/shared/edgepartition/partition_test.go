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
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/controlplane/shared/edgepartition"
)

func TestIntIDToHex(t *testing.T) {
	tests := []struct {
		input    int
		expected string
	}{
		{0, "00"},
		{10, "0a"},
		{255, "ff"},
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
			pCount:   16,
			pID:      0,
			expected: []string{"00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "0a", "0b", "0c", "0d", "0e", "0f"},
		},
		{
			pCount:   16,
			pID:      1,
			expected: []string{"10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "1a", "1b", "1c", "1d", "1e", "1f"},
		},
		{
			pCount:   16,
			pID:      15,
			expected: []string{"f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "fa", "fb", "fc", "fd", "fe", "ff"},
		},
		{
			pCount:   155,
			pID:      0,
			expected: []string{"00"},
		},
		{
			pCount:   155,
			pID:      1,
			expected: []string{"01", "02"},
		},
		{
			pCount:   155,
			pID:      2,
			expected: []string{"03"},
		},
		{
			pCount:   155,
			pID:      154,
			expected: []string{"fe", "ff"},
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
	expected := "6b"

	actual := edgepartition.EdgeIDToPartition(id)
	assert.Equal(t, expected, actual)
}

func TestEdgeToCPNATSTopic(t *testing.T) {
	id, _ := uuid.FromString("6ba7b810-9dad-11d1-80b4-00c04fd430c8") // Example UUID

	tests := []struct {
		name     string
		expected string
		topic    corepb.EdgeCPTopic
		durable  bool
	}{
		{
			name:     "status",
			expected: "e2cp.6b.6ba7b810-9dad-11d1-80b4-00c04fd430c8.status",
			topic:    corepb.EDGE_CP_TOPIC_STATUS,
			durable:  false,
		},
		{
			name:     "Durablestatus",
			expected: "e2cp.6b.6ba7b810-9dad-11d1-80b4-00c04fd430c8.Durablestatus",
			topic:    corepb.EDGE_CP_TOPIC_STATUS,
			durable:  true,
		},
		{
			name:     "video",
			expected: "e2cp.6b.6ba7b810-9dad-11d1-80b4-00c04fd430c8.video",
			topic:    corepb.EDGE_CP_TOPIC_VIDEO,
			durable:  false,
		},
		{
			name:     "Durablevideo",
			expected: "e2cp.6b.6ba7b810-9dad-11d1-80b4-00c04fd430c8.Durablevideo",
			topic:    corepb.EDGE_CP_TOPIC_VIDEO,
			durable:  true,
		},
	}

	for _, test := range tests {
		actual, err := edgepartition.EdgeToCPNATSTopic(id, test.topic, test.durable)

		require.NoError(t, err)
		assert.Equal(t, test.expected, actual)
	}
}

func TestEdgeToCPNATSPartitionTopic(t *testing.T) {
	partition := "6b"

	tests := []struct {
		name     string
		expected string
		topic    corepb.EdgeCPTopic
		durable  bool
	}{
		{
			name:     "status",
			expected: "e2cp.6b.*.status",
			topic:    corepb.EDGE_CP_TOPIC_STATUS,
			durable:  false,
		},
		{
			name:     "Durablestatus",
			expected: "e2cp.6b.*.Durablestatus",
			topic:    corepb.EDGE_CP_TOPIC_STATUS,
			durable:  true,
		},
		{
			name:     "video",
			expected: "e2cp.6b.*.video",
			topic:    corepb.EDGE_CP_TOPIC_VIDEO,
			durable:  false,
		},
		{
			name:     "Durablevideo",
			expected: "e2cp.6b.*.Durablevideo",
			topic:    corepb.EDGE_CP_TOPIC_VIDEO,
			durable:  true,
		},
	}

	for _, test := range tests {
		actual, err := edgepartition.EdgeToCPNATSPartitionTopic(partition, test.topic, test.durable)

		require.NoError(t, err)
		assert.Equal(t, test.expected, actual)
	}
}

func TestCPToEdgeNATSTopicBase(t *testing.T) {
	id, _ := uuid.FromString("6ba7b810-9dad-11d1-80b4-00c04fd430c8") // Example UUID
	actual, err := edgepartition.CPToEdgeNATSTopicBase(id)
	expected := "cp2e.6b.6ba7b810-9dad-11d1-80b4-00c04fd430c8"

	require.NoError(t, err)
	assert.Equal(t, expected, actual)
}

func TestCPToEdgeNATSTopic(t *testing.T) {
	id, _ := uuid.FromString("6ba7b810-9dad-11d1-80b4-00c04fd430c8") // Example UUID
	actual, err := edgepartition.CPToEdgeNATSTopic(id, corepb.CP_EDGE_TOPIC_STATUS, false)
	expected := "cp2e.6b.6ba7b810-9dad-11d1-80b4-00c04fd430c8.status"

	require.NoError(t, err)
	assert.Equal(t, expected, actual)
}

func TestCPNATSTopic(t *testing.T) {
	id, _ := uuid.FromString("6ba7b810-9dad-11d1-80b4-00c04fd430c8") // Example UUID
	actual, err := edgepartition.CPNATSTopic(edgepartition.EdgeIDToPartition(id), id.String(), corepb.CP_TOPIC_DEVICE_UPDATE, true)
	expected := "cp2cp.6b.6ba7b810-9dad-11d1-80b4-00c04fd430c8.DurabledeviceUpdate"

	require.NoError(t, err)
	assert.Equal(t, expected, actual)
}
