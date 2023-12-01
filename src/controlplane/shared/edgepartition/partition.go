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

// Package edgepartition has utilities to work with NATS static partition for our edge devices.
// Our edge devices are partition as follows:
// (edge2cp|cp2edge).<org_id>.<fleet_partition_id>.<fleet_id>.<device_id>.<channel_name>
// Each of org_if, fleet_id, device_id are all UUIDs.
// The fleet_partition_id is the first 4 hex characters of the fleet_id. This allows us to have at most
// 4096 partitions.
package edgepartition

import (
	"fmt"
	"log"
	"math"

	"github.com/gofrs/uuid/v5"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
)

const (
	// KeySpaceSize is the number of hex characters to use for the keyspace size.
	KeySpaceSize   = 3
	edgeToCPPrefix = "e2cp"
	cpToEdgePrefix = "cp2e"
	cpToCpPrefix   = "cp2cp"
)

func init() {
	pflag.Int("partition_count", -1, "The total number of partition. The keyspace is evenly divided by the partition_count.")
	pflag.Int("partition_id", 0, "The ID Of the partition. This subscribes to the segment of the partition defined by the ID.")
}

func IntIDToHex(id int) string {
	return fmt.Sprintf("%0*x", KeySpaceSize, id)
}

func minPartID() int {
	partitionCount := viper.GetInt("partition_count")
	partitionID := viper.GetInt("partition_id")
	totalKeys := int(math.Pow(16, KeySpaceSize))
	return partitionID * totalKeys / partitionCount
}

func maxPartID() int {
	partitionCount := viper.GetInt("partition_count")
	partitionID := viper.GetInt("partition_id")

	totalKeys := int(math.Pow(16, KeySpaceSize))
	if partitionID+1 == partitionCount {
		return totalKeys
	}
	return (partitionID + 1) * totalKeys / partitionCount
}

// GenerateRange produces values between min/max in hex.
func GenerateRange() []string {
	pc := viper.GetInt("partition_count")
	if pc < 0 {
		log.Fatal("Must set a positive partition count for partitioned services.")
	}
	mn := minPartID()
	mx := maxPartID()
	r := make([]string, mx-mn)
	for i := mn; i < mx; i++ {
		r[i-mn] = IntIDToHex(i)
	}
	return r
}

func EdgeIDToPartition(id uuid.UUID) string {
	return id.String()[0:KeySpaceSize]
}

func EdgeToCPNATSTopic(edgeID uuid.UUID, topic corepb.EdgeCPTopic, isDurable bool) (string, error) {
	return edgeToCPNATSTopic(EdgeIDToPartition(edgeID), edgeID.String(), topic, isDurable)
}

func CPToEdgeNATSTopicBase(edgeID uuid.UUID) (string, error) {
	return fmt.Sprintf("%s.%s.%s", cpToEdgePrefix, EdgeIDToPartition(edgeID), edgeID.String()), nil
}

func CPToEdgeNATSTopic(edgeID uuid.UUID, topic corepb.CPEdgeTopic, isDurable bool) (string, error) {
	gen := func(str string) string {
		if isDurable {
			str = "Durable" + str
		}

		base, _ := CPToEdgeNATSTopicBase(edgeID)
		return fmt.Sprintf("%s.%s", base, str)
	}
	switch topic {
	case corepb.CP_EDGE_TOPIC_STATUS:
		return gen("status"), nil
	case corepb.CP_EDGE_TOPIC_VIDEO:
		return gen("video"), nil
	case corepb.CP_EDGE_TOPIC_EXEC:
		return gen("exec"), nil
	case corepb.CP_EDGE_TOPIC_METRICS:
		return gen("metrics"), nil
	case corepb.CP_EDGE_TOPIC_FILE_TRANSFER:
		return gen("filetransfer"), nil
	default:
		return "", fmt.Errorf("bad topic %s", topic.String())
	}
}

func edgeToCPNATSTopic(partition string, edgeID string, topic corepb.EdgeCPTopic, isDurable bool) (string, error) {
	gen := func(str string) string {
		if isDurable {
			str = "Durable" + str
		}
		return fmt.Sprintf("%s.%s.%s.%s", edgeToCPPrefix, partition, edgeID, str)
	}
	switch topic {
	case corepb.EDGE_CP_TOPIC_STATUS:
		return gen("status"), nil
	case corepb.EDGE_CP_TOPIC_VIDEO:
		return gen("video"), nil
	case corepb.EDGE_CP_TOPIC_EXEC:
		return gen("exec"), nil
	case corepb.EDGE_CP_TOPIC_METRICS:
		return gen("metrics"), nil
	case corepb.EDGE_CP_TOPIC_FILE_TRANSFER:
		return gen("filetransfer"), nil
	default:
		return "", fmt.Errorf("bad topic %s", topic.String())
	}
}

func EdgeToCPNATSPartitionTopic(partition string, topic corepb.EdgeCPTopic, isDurable bool) (string, error) {
	return edgeToCPNATSTopic(partition, "*", topic, isDurable)
}

// CPNATSTopic is for topics published and read in the controlplane, but is still partitioned.
func CPNATSTopic(partition string, edgeID string, topic corepb.CPTopic, isDurable bool) (string, error) {
	gen := func(str string) string {
		if isDurable {
			str = "Durable" + str
		}
		return fmt.Sprintf("%s.%s.%s.%s", cpToCpPrefix, partition, edgeID, str)
	}
	switch topic {
	case corepb.CP_TOPIC_DEVICE_CONNECTED:
		return gen("deviceConnected"), nil
	case corepb.CP_TOPIC_PHYSICAL_PIPELINE_RECONCILIATION:
		return gen("physicalPipelineReconciliation"), nil
	default:
		return "", fmt.Errorf("bad topic %s", topic.String())
	}
}

func CPNATSPartitionTopic(partition string, topic corepb.CPTopic, isDurable bool) (string, error) {
	return CPNATSTopic(partition, "*", topic, isDurable)
}
