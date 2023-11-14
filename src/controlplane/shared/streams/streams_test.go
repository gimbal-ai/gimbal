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

package streams_test

import (
	"bytes"
	"context"
	"testing"
	"time"

	"github.com/gofrs/uuid/v5"
	"github.com/nats-io/nats.go/jetstream"
	"github.com/spf13/viper"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/controlplane/shared/edgepartition"
	"gimletlabs.ai/gimlet/src/controlplane/shared/streams"
	"gimletlabs.ai/gimlet/src/shared/services/msgbus"
	"gimletlabs.ai/gimlet/src/shared/services/natstest"
)

func TestJetStream_AddStreams(t *testing.T) {
	viper.Set("jetstream_cluster_size", 1)

	nc, cleanup := natstest.MustStartTestNATS(t)
	defer cleanup()

	js := streams.MustConnectCPJetStream(nc)

	deviceID := uuid.Must(uuid.NewV4())

	topic, err := edgepartition.EdgeToCPNATSPartitionTopic(edgepartition.EdgeIDToPartition(deviceID), corepb.EDGE_CP_TOPIC_METRICS, true)
	require.NoError(t, err)
	subTopic, err := edgepartition.EdgeToCPNATSTopic(deviceID, corepb.EDGE_CP_TOPIC_METRICS, true)
	require.NoError(t, err)

	s, err := msgbus.NewJetStreamStreamer(js, "metrics")
	require.NoError(t, err)

	// Publish data to the subject.
	require.NoError(t, s.Publish(subTopic, []byte("123")))

	ctx := context.Background()
	_, err = js.CreateOrUpdateConsumer(ctx, "metrics", jetstream.ConsumerConfig{
		Durable:       msgbus.FormatConsumerName(topic, "processor"),
		FilterSubject: topic,
		DeliverPolicy: jetstream.DeliverAllPolicy,
		AckWait:       30 * time.Second,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxAckPending: 50,
	})
	require.NoError(t, err)
	ch1 := make(chan jetstream.Msg)
	pSub, err := s.PersistentSubscribe(topic, "processor", func(m jetstream.Msg) {
		ch1 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)

	// Should receive all messages that were published.
	func() {
		for {
			select {
			case m := <-ch1:
				assert.True(t, bytes.Equal([]byte("123"), m.Data()))
				return
			case <-time.After(50 * time.Millisecond):
				assert.False(t, true)
				return
			}
		}
	}()

	pSub.Close()
}

func TestJetStream_InitializeConsumers(t *testing.T) {
	viper.Set("jetstream_cluster_size", 1)

	nc, cleanup := natstest.MustStartTestNATS(t)
	defer cleanup()

	js := streams.MustConnectCPJetStream(nc)

	// Create some existing consumers.
	streamName := "metrics"
	svcName := "metricsProcessor"

	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err := js.CreateConsumer(ctx, streamName, jetstream.ConsumerConfig{
		Durable:       "metrics__|metricsProcessor",
		FilterSubject: "metrics.*",
	})
	require.NoError(t, err)

	_, err = js.CreateConsumer(ctx, streamName, jetstream.ConsumerConfig{
		Durable:       "metrics__|otherService",
		FilterSubject: "metrics.*",
	})
	require.NoError(t, err)

	_, err = js.CreateConsumer(ctx, streamName, jetstream.ConsumerConfig{
		Durable:       "metrics___12__|metricsProcessor",
		FilterSubject: "metrics.*.12.*",
	})
	require.NoError(t, err)

	consumers := []*streams.CPConsumer{
		// Existing stream.
		{
			Subject: "metrics.*",
		},
		// New stream.
		{
			Subject: "metrics.*.13.*",
		},
	}
	err = streams.InitializeConsumers(js, svcName, streamName, consumers)
	require.NoError(t, err)

	stream, err := js.Stream(context.Background(), streamName)
	require.NoError(t, err)
	consumerList := stream.ListConsumers(ctx)
	consumerNames := make([]string, 0)
	for c := range consumerList.Info() {
		consumerNames = append(consumerNames, c.Name)
	}

	assert.ElementsMatch(t, []string{"metrics__|metricsProcessor", "metrics__|otherService", "metrics___13__|metricsProcessor"}, consumerNames)
}
