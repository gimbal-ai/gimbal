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
	"context"
	"sync"
	"testing"
	"time"

	"github.com/gofrs/uuid/v5"
	"github.com/gogo/protobuf/proto"
	"github.com/gogo/protobuf/types"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"go.uber.org/mock/gomock"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/controlplane/shared/edgepartition"
	"gimletlabs.ai/gimlet/src/shared/services/msgbus"
	"gimletlabs.ai/gimlet/src/shared/services/natstest"
	utils "gimletlabs.ai/gimlet/src/shared/uuidutils"
)

func TestDurablePartitionHandler_MessageHandler(t *testing.T) {
	viper.Set("partition_id", 0)
	viper.Set("partition_count", 1)

	ctrl := gomock.NewController(t)
	defer ctrl.Finish()

	nc, natsCleanup := natstest.MustStartTestNATS(t)
	defer natsCleanup()

	deviceID := uuid.Must(uuid.NewV4())

	js := msgbus.MustConnectJetStream(nc)

	subject, err := edgepartition.EdgeToCPNATSPartitionTopic("*", corepb.EDGE_CP_TOPIC_STATUS, true)
	require.NoError(t, err)
	_, err = js.CreateStream(context.Background(), jetstream.StreamConfig{
		Name:     "status",
		Subjects: []string{subject},
		MaxAge:   2 * time.Minute,
		Replicas: 1,
		Storage:  jetstream.MemoryStorage,
	})
	require.NoError(t, err)

	ctx := context.Background()
	partitions := edgepartition.GenerateRange()
	for _, p := range partitions {
		sub, err := edgepartition.EdgeToCPNATSPartitionTopic(p, corepb.EDGE_CP_TOPIC_STATUS, true)
		require.NoError(t, err)
		_, err = js.CreateOrUpdateConsumer(ctx, "status", jetstream.ConsumerConfig{
			Durable:       msgbus.FormatConsumerName(sub, "testsvc"),
			FilterSubject: sub,
			DeliverPolicy: jetstream.DeliverAllPolicy,
			AckWait:       30 * time.Second,
			AckPolicy:     jetstream.AckExplicitPolicy,
			MaxAckPending: 50,
		})
		require.NoError(t, err)
	}

	s, err := msgbus.NewJetStreamStreamer(js, "status")
	require.NoError(t, err)

	var wg sync.WaitGroup

	wg.Add(1)
	handler := func(md *corepb.EdgeCPMetadata, anyMsg *types.Any) error {
		assert.Equal(t, corepb.EDGE_CP_TOPIC_STATUS, md.Topic)
		assert.Equal(t, utils.ProtoFromUUID(deviceID), md.DeviceID)
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
}
