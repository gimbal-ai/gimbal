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
	"context"
	"time"

	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
	"golang.org/x/sync/errgroup"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/controlplane/shared/edgepartition"
	"gimletlabs.ai/gimlet/src/shared/services/msgbus"
)

func init() {
	pflag.Int("jetstream_cluster_size", 3, "The number of JetStream replicas, used to configure stream replication.")
}

var defaultConsumerConfig = jetstream.ConsumerConfig{
	DeliverPolicy: jetstream.DeliverAllPolicy,
	AckWait:       30 * time.Second,
	AckPolicy:     jetstream.AckExplicitPolicy,
	MaxAckPending: 50,
}

var DurableStreamTopics = []corepb.EdgeCPTopic{
	corepb.EDGE_CP_TOPIC_METRICS,
}

var EdgeCPTopicToStreamName = map[corepb.EdgeCPTopic]string{
	corepb.EDGE_CP_TOPIC_METRICS: "metrics",
}

// MustConnectCPJetStream creates a new JetStream connection.
func MustConnectCPJetStream(nc *nats.Conn) jetstream.JetStream {
	js := msgbus.MustConnectJetStream(nc)

	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Minute)
	defer cancel()

	streams := getDurableStreams()
	for _, s := range streams {
		_, err := js.CreateStream(ctx, s)
		if err != nil {
			log.WithError(err).Fatal("Could not start up durable streams")
		}
	}

	return js
}

func getDurableStreams() []jetstream.StreamConfig {
	clusterSize := viper.GetInt("jetstream_cluster_size")

	metricsTopic, err := edgepartition.EdgeToCPNATSPartitionTopic("*", corepb.EDGE_CP_TOPIC_METRICS, true)
	if err != nil {
		log.WithError(err).Fatal("Could not get metrics topic")
	}

	return []jetstream.StreamConfig{
		{
			Name:     EdgeCPTopicToStreamName[corepb.EDGE_CP_TOPIC_METRICS],
			MaxAge:   15 * time.Minute,
			Replicas: clusterSize,
			Subjects: []string{metricsTopic},
		},
	}
}

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
			config = defaultConsumerConfig
		} else {
			config = *c.Config
		}
		config.Durable = msgbus.FormatConsumerName(c.Subject, serviceName)
		config.FilterSubject = c.Subject

		// Track which existing consumers should still exist.
		updatedConsumerMap[config.Durable] = true

		createGroup.Go(func() error {
			_, err = js.CreateOrUpdateConsumer(ctx, streamName, config)
			if err != nil {
				return err
			}
			return nil
		})
	}

	if err := createGroup.Wait(); err != nil {
		return err
	}

	return nil
}

func InitializeConsumersForPartition(consumerName string, js jetstream.JetStream, topic corepb.EdgeCPTopic, config *jetstream.ConsumerConfig) error {
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

	err := InitializeConsumers(js, consumerName, EdgeCPTopicToStreamName[topic], consumers)
	if err != nil {
		return err
	}

	return nil
}
