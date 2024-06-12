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
	"context"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	log "github.com/sirupsen/logrus"
	"golang.org/x/sync/errgroup"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/controlplane/shared/edgepartition"
	"gimletlabs.ai/gimlet/src/controlplane/shared/streams"
	"gimletlabs.ai/gimlet/src/shared/services/msgbus"
)

// CPConsumer is a controlplane consumer of a durable stream.
type CPConsumer struct {
	// The subject to listen to.
	Subject string
	// The config to use, will use default if none specified.
	Config *jetstream.ConsumerConfig
}

// InitializeConsumers must be called by the service to ensure all consumers are set up for a stream before use.
// It ensures that the available consumers match the consumers the service expects.
func InitializeConsumers(js jetstream.JetStream, serviceName string, streamName string, consumers []*CPConsumer) error {
	// Get all consumers currently setup for this stream.
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	stream, err := js.Stream(ctx, streamName)
	if err != nil {
		return err
	}
	consumerList := stream.ListConsumers(ctx)

	// Filter through consumers for this service.
	updatedConsumerMap := make(map[string]bool)
	for c := range consumerList.Info() {
		svcName, err := msgbus.GetPersistentNameFromConsumerName(c.Name)
		if err != nil {
			log.WithField("stream", streamName).WithField("consumer", c.Name).WithError(err).Error("invalid consumer name in stream")
			continue
		}
		if svcName != serviceName {
			continue
		}
		updatedConsumerMap[c.Name] = false
	}

	// Update or create any new consumers.
	ctx, cancel = context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	createGroup, ctx := errgroup.WithContext(ctx)

	for _, c := range consumers {
		var config jetstream.ConsumerConfig
		if c.Config == nil {
			config = jetstream.ConsumerConfig{
				DeliverPolicy: jetstream.DeliverAllPolicy,
				AckWait:       30 * time.Second,
				AckPolicy:     jetstream.AckExplicitPolicy,
				MaxAckPending: 50,
				Replicas:      1,
			}
		} else {
			config = *c.Config
		}
		config.Durable = msgbus.FormatConsumerName(c.Subject, serviceName)
		config.FilterSubject = c.Subject

		// Track which existing consumers should still exist.
		updatedConsumerMap[config.Durable] = true

		createGroup.Go(func() error {
			var err error
			_, err = js.CreateOrUpdateConsumer(ctx, streamName, config)
			if err != nil {
				return err
			}
			return nil
		})
	}

	return createGroup.Wait()
}

// InitializeConsumersForEdgeToCPPartition to initialize jetstream consumers for an edge to controlplane partition.
func InitializeConsumersForEdgeToCPPartition(consumerName string, js jetstream.JetStream, topic corepb.EdgeCPTopic, config *jetstream.ConsumerConfig) error {
	consumers := make([]*CPConsumer, 0)
	partitions := edgepartition.GenerateRange()
	for _, p := range partitions {
		sub, err := edgepartition.EdgeToCPNATSPartitionTopic(p, topic, true)
		if err != nil {
			return err
		}

		consumers = append(consumers, &CPConsumer{
			Subject: sub,
			Config:  config,
		})
	}

	err := InitializeConsumers(js, consumerName, streams.EdgeCPTopicToStreamName[topic], consumers)
	if err != nil {
		return err
	}

	return nil
}

// InitializeConsumersForCPPartition to initialize jetstream consumers for a controlplane partition.
func InitializeConsumersForCPPartition(consumerName string, js jetstream.JetStream, topic corepb.CPTopic, config *jetstream.ConsumerConfig) error {
	consumers := make([]*CPConsumer, 0)
	partitions := edgepartition.GenerateRange()
	for _, p := range partitions {
		sub, err := edgepartition.CPNATSPartitionTopic(p, topic, true)
		if err != nil {
			return err
		}

		consumers = append(consumers, &CPConsumer{
			Subject: sub,
			Config:  config,
		})
	}

	err := InitializeConsumers(js, consumerName, streams.CPTopicToStreamName[topic], consumers)
	if err != nil {
		return err
	}

	return nil
}

func InitializeCPStream(js jetstream.JetStream, topic corepb.CPTopic, durable bool) error {
	subject, err := edgepartition.CPNATSPartitionTopic("*", topic, durable)
	if err != nil {
		return err
	}
	_, err = js.CreateStream(context.Background(), jetstream.StreamConfig{
		Name:     streams.CPTopicToStreamName[topic],
		Subjects: []string{subject},
		MaxAge:   2 * time.Minute,
		Replicas: 1,
		Storage:  jetstream.MemoryStorage,
	})
	return err
}

func InitializeEdgeToCPStream(js jetstream.JetStream, topic corepb.EdgeCPTopic, durable bool) error {
	subject, err := edgepartition.EdgeToCPNATSPartitionTopic("*", topic, durable)
	if err != nil {
		return err
	}
	_, err = js.CreateStream(context.Background(), jetstream.StreamConfig{
		Name:     streams.EdgeCPTopicToStreamName[topic],
		Subjects: []string{subject},
		MaxAge:   2 * time.Minute,
		Replicas: 1,
		Storage:  jetstream.MemoryStorage,
	})
	return err
}

type MockJetstreamMsg struct {
	MsgData     []byte
	MsgSubject  string
	MsgMetadata *jetstream.MsgMetadata
	nakCalled   bool
}

func (m *MockJetstreamMsg) Metadata() (*jetstream.MsgMetadata, error) {
	return m.MsgMetadata, nil
}

func (*MockJetstreamMsg) Headers() nats.Header {
	return nil
}

func (m *MockJetstreamMsg) Data() []byte {
	return m.MsgData
}

func (m *MockJetstreamMsg) Subject() string {
	return m.MsgSubject
}

func (*MockJetstreamMsg) Reply() string {
	return ""
}

func (*MockJetstreamMsg) Ack() error {
	return nil
}

func (*MockJetstreamMsg) DoubleAck(context.Context) error {
	return nil
}

func (m *MockJetstreamMsg) Nak() error {
	m.nakCalled = true

	return nil
}

func (m *MockJetstreamMsg) NakWithDelay(time.Duration) error {
	m.nakCalled = true

	return nil
}

func (*MockJetstreamMsg) InProgress() error {
	return nil
}

func (*MockJetstreamMsg) Term() error {
	return nil
}

func (*MockJetstreamMsg) TermWithReason(string) error {
	return nil
}

func (m *MockJetstreamMsg) NakCalled() bool {
	return m.nakCalled
}
