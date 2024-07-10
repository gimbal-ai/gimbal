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

package edgepartition_test

import (
	"context"
	"fmt"
	"sync"
	"testing"

	"github.com/gofrs/uuid/v5"
	"github.com/gogo/protobuf/proto"
	"github.com/gogo/protobuf/types"
	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/controlplane/shared/edgepartition"
	"gimletlabs.ai/gimlet/src/controlplane/shared/streams"
	"gimletlabs.ai/gimlet/src/shared/services/msgbus"
	"gimletlabs.ai/gimlet/src/shared/services/natstest"
	"gimletlabs.ai/gimlet/src/shared/testing/testutils"
	utils "gimletlabs.ai/gimlet/src/shared/uuidutils"
)

func setupTest(t *testing.T) (*nats.Conn, jetstream.JetStream) {
	viper.Set("partition_id", 0)
	viper.Set("partition_count", 1)
	nc := natstest.MustStartTestNATS(t)

	js := msgbus.MustConnectJetStream(nc)

	err := testutils.InitializeEdgeToCPStream(js, corepb.EDGE_CP_TOPIC_STATUS, true)
	require.NoError(t, err)

	err = testutils.InitializeConsumersForEdgeToCPPartition("testsvc", js, corepb.EDGE_CP_TOPIC_STATUS, nil)
	require.NoError(t, err)
	return nc, js
}

func TestDurablePartitionHandler_MessageHandler(t *testing.T) {
	nc, js := setupTest(t)
	deviceID := uuid.Must(uuid.NewV4())
	// Test normal operation.
	var wg sync.WaitGroup
	s, err := msgbus.NewJetStreamStreamer(js, streams.EdgeCPTopicToStreamName[corepb.EDGE_CP_TOPIC_STATUS])
	require.NoError(t, err)

	wg.Add(1)
	handler := func(_ context.Context, md *edgepartition.MsgMetadata, anyMsg *types.Any) error {
		assert.Equal(t, corepb.EDGE_CP_TOPIC_STATUS, md.EdgeCPMetadata.Topic)
		assert.Equal(t, utils.ProtoFromUUID(deviceID), md.EdgeCPMetadata.DeviceID)
		hbMsg := &corepb.EdgeHeartbeat{}
		err := types.UnmarshalAny(anyMsg, hbMsg)
		require.NoError(t, err)
		assert.Equal(t, int64(10), hbMsg.SeqID)
		wg.Done()
		return nil
	}

	handlers := make(map[string]edgepartition.DurableMessageHandler)
	handlers[proto.MessageName(&corepb.EdgeHeartbeat{})] = handler
	ph := edgepartition.NewDurablePartitionHandler(s, "testsvc", handlers).WithEdgeToCPTopic(corepb.EDGE_CP_TOPIC_STATUS)
	err = ph.Start()
	require.NoError(t, err)

	anyMsg, err := types.MarshalAny(&corepb.EdgeHeartbeat{SeqID: 10})
	require.NoError(t, err)
	msg := &corepb.EdgeCPMessage{
		Metadata: &corepb.EdgeCPMetadata{
			Topic:         corepb.EDGE_CP_TOPIC_STATUS,
			DeviceID:      utils.ProtoFromUUID(deviceID),
			RecvTimestamp: nil,
		},
		Msg: anyMsg,
	}
	b, err := msg.Marshal()
	require.NoError(t, err)

	topic, err := edgepartition.EdgeToCPNATSTopic(deviceID, corepb.EDGE_CP_TOPIC_STATUS, true)
	require.NoError(t, err)
	err = nc.Publish(topic, b)
	require.NoError(t, err)

	wg.Wait()
	ph.Stop()
}

func TestDurablePartitionHandler_MessageHandlerError(t *testing.T) {
	nc, js := setupTest(t)
	deviceID := uuid.Must(uuid.NewV4())
	var wg sync.WaitGroup
	s, err := msgbus.NewJetStreamStreamer(js, streams.EdgeCPTopicToStreamName[corepb.EDGE_CP_TOPIC_STATUS])
	require.NoError(t, err)

	// Channel to track the sequence of messages received.
	// Has a buffer so we don't block the handler.
	seqCh := make(chan int64, 2)

	wg.Add(2)
	handler := func(_ context.Context, md *edgepartition.MsgMetadata, anyMsg *types.Any) error {
		defer func() {
			wg.Done()
		}()
		assert.Equal(t, corepb.EDGE_CP_TOPIC_STATUS, md.EdgeCPMetadata.Topic)
		assert.Equal(t, utils.ProtoFromUUID(deviceID), md.EdgeCPMetadata.DeviceID)
		hbMsg := &corepb.EdgeHeartbeat{}
		err := types.UnmarshalAny(anyMsg, hbMsg)
		require.NoError(t, err)
		seqCh <- hbMsg.SeqID
		// Error out on the first call to the handler.
		if hbMsg.SeqID == 10 {
			return fmt.Errorf("error")
		}
		assert.Equal(t, int64(11), hbMsg.SeqID)
		return nil
	}

	handlers := make(map[string]edgepartition.DurableMessageHandler)
	handlers[proto.MessageName(&corepb.EdgeHeartbeat{})] = handler
	ph := edgepartition.NewDurablePartitionHandler(s, "testsvc", handlers).WithEdgeToCPTopic(corepb.EDGE_CP_TOPIC_STATUS)
	err = ph.Start()
	require.NoError(t, err)

	topic, err := edgepartition.EdgeToCPNATSTopic(deviceID, corepb.EDGE_CP_TOPIC_STATUS, true)
	require.NoError(t, err)

	anyMsg, err := types.MarshalAny(&corepb.EdgeHeartbeat{SeqID: 10})
	require.NoError(t, err)
	msg := &corepb.EdgeCPMessage{
		Metadata: &corepb.EdgeCPMetadata{
			Topic:         corepb.EDGE_CP_TOPIC_STATUS,
			DeviceID:      utils.ProtoFromUUID(deviceID),
			RecvTimestamp: nil,
		},
		Msg: anyMsg,
	}
	b, err := msg.Marshal()
	require.NoError(t, err)
	err = nc.Publish(topic, b)
	require.NoError(t, err)
	anyMsg, err = types.MarshalAny(&corepb.EdgeHeartbeat{SeqID: 11})
	require.NoError(t, err)
	msg = &corepb.EdgeCPMessage{
		Metadata: &corepb.EdgeCPMetadata{
			Topic:         corepb.EDGE_CP_TOPIC_STATUS,
			DeviceID:      utils.ProtoFromUUID(deviceID),
			RecvTimestamp: nil,
		},
		Msg: anyMsg,
	}
	b, err = msg.Marshal()
	require.NoError(t, err)

	err = nc.Publish(topic, b)
	require.NoError(t, err)

	wg.Wait()
	ph.Stop()
	assert.Equal(t, int64(10), <-seqCh)
	assert.Equal(t, int64(11), <-seqCh)
}
