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
	corepb.CP_TOPIC_DEVICE_UPDATE:                      "deviceUpdate",
	corepb.CP_TOPIC_DEVICE_DISCONNECTED:                "deviceDisconnected",
	corepb.CP_TOPIC_PIPELINE_DEPLOYMENT_RECONCILIATION: "pipelineDeploymentReconciliation",
}

// MustConnectCPJetStream creates a new JetStream connection.
func MustConnectCPJetStream(nc *nats.Conn) jetstream.JetStream {
	return msgbus.MustConnectJetStream(nc)
}
