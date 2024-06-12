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

package streams

import (
	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/spf13/pflag"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/shared/services/msgbus"
)

func init() {
	pflag.Int("jetstream_cluster_size", 3, "The number of JetStream replicas, used to configure stream replication.")
}

var IsEdgeCPTopicDurable = map[corepb.EdgeCPTopic]bool{
	corepb.EDGE_CP_TOPIC_METRICS: true,
	corepb.EDGE_CP_TOPIC_EXEC:    true,
	corepb.EDGE_CP_TOPIC_INFO:    true,
}

var EdgeCPTopicToStreamName = map[corepb.EdgeCPTopic]string{
	corepb.EDGE_CP_TOPIC_METRICS: "metrics",
	corepb.EDGE_CP_TOPIC_EXEC:    "exec",
	corepb.EDGE_CP_TOPIC_INFO:    "info",
	corepb.EDGE_CP_TOPIC_STATUS:  "status", // Used for tests.
}

var CPTopicToStreamName = map[corepb.CPTopic]string{
	corepb.CP_TOPIC_DEVICE_CONNECTED:                 "deviceConnected",
	corepb.CP_TOPIC_DEVICE_DISCONNECTED:              "deviceDisconnected",
	corepb.CP_TOPIC_PHYSICAL_PIPELINE_RECONCILIATION: "physicalPipelineReconciliation",
}

// MustConnectCPJetStream creates a new JetStream connection.
func MustConnectCPJetStream(nc *nats.Conn) jetstream.JetStream {
	return msgbus.MustConnectJetStream(nc)
}
