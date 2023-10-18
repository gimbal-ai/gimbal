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
	"math"

	"github.com/gofrs/uuid/v5"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

const (
	// KeySpaceSize is the number of hex characters to use for the keyspace size.
	KeySpaceSize = 3
)

func init() {
	pflag.Int("partition_count", 2, "The total number of partition. The keyspace is evenly divided by the partition_count.")
	pflag.Int("partition_id", 0, "The ID Of the partition. This subscribes to the segment of the partition defined by the ID.")
}

func IntIDToHex(id int) string {
	return fmt.Sprintf("%0*x", KeySpaceSize, id)
}

func minPartID() int {
	partitionCount := viper.GetInt("partition_count")
	partitionID := viper.GetInt("partition_id")
	totalKeys := int(math.Pow(16, KeySpaceSize))
	return partitionID * (totalKeys / partitionCount)
}

func maxPartID() int {
	partitionCount := viper.GetInt("partition_count")
	partitionID := viper.GetInt("partition_id")

	totalKeys := int(math.Pow(16, KeySpaceSize))
	return (partitionID + 1) * (totalKeys / partitionCount)
}

// GenerateRange produces values between min/max in hex.
func GenerateRange() []string {
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
