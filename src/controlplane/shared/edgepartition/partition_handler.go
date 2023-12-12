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

package edgepartition

import (
	"sync"

	"github.com/gogo/protobuf/types"
	"github.com/nats-io/nats.go"
	log "github.com/sirupsen/logrus"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
)

const (
	channelSize = 4096
)

type MessageHandler interface {
	HandleMessage(*corepb.EdgeCPMetadata, *types.Any, string)
}

type MessageHandlerFactory func() MessageHandler

type MessageHandlerConfig struct {
	// The max number of goroutines which should be used to process the messages for this partition.
	NumGoroutines int64
}

// PartitionHandler handles any incoming NATS messages from an edge device.
type PartitionHandler struct {
	nc *nats.Conn

	// Signal used to quit.
	done chan struct{}
	once sync.Once
}

// NewPartitionHandler creates a new partition handler for the given topic.
func NewPartitionHandler(nc *nats.Conn) *PartitionHandler {
	return &PartitionHandler{nc: nc, done: make(chan struct{})}
}

// RegisterHandler registers a new message handler for a topic.
func (p *PartitionHandler) RegisterHandler(e2cpTopic corepb.EdgeCPTopic, factory MessageHandlerFactory, config *MessageHandlerConfig) error {
	partitions := GenerateRange()
	log.WithFields(log.Fields{
		"min_partition_id": partitions[0],
		"max_partition_id": partitions[len(partitions)-1],
	}).Info("Subscribing to NATS")
	for _, pr := range partitions {
		subCh := make(chan *nats.Msg, channelSize)
		err := p.startPartitionHandler(pr, subCh, e2cpTopic, factory, config)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *PartitionHandler) startPartitionHandler(partition string, subCh chan *nats.Msg, e2cpTopic corepb.EdgeCPTopic, factory MessageHandlerFactory, config *MessageHandlerConfig) error {
	topic, err := EdgeToCPNATSPartitionTopic(partition, e2cpTopic, false)
	if err != nil {
		return err
	}
	topicLog := log.WithField("topic", topic)

	natsSub, err := p.nc.ChanSubscribe(topic, subCh)
	if err != nil {
		topicLog.WithError(err).Error("Failed to subscribe to NATS")
		return err
	}

	// The channel which workers will dequeue messages from.
	workerCh := make(chan *corepb.EdgeCPMessage, channelSize)

	go func() {
		for {
			select {
			case <-p.done:
				err := natsSub.Unsubscribe()
				if err != nil {
					topicLog.WithError(err).Error("Failed to unsubscribe")
				}
				return
			case msg := <-subCh:
				e2CpMsg := &corepb.EdgeCPMessage{}
				err = e2CpMsg.Unmarshal(msg.Data)
				if err != nil {
					topicLog.WithError(err).Error("Failed to unmarshal CPEdgeMessage... Skipping.")
					continue
				}
				// Put message on message handling queue. We don't want to block the NATS reading goroutine, or else
				// NATS will begin dropping messages.
				workerCh <- e2CpMsg
			}
		}
	}()

	// Start up workers.
	n := 0
	for n < int(config.NumGoroutines) {
		p.partitionWorker(topicLog, factory(), workerCh)
		n++
	}

	return nil
}

func (p *PartitionHandler) partitionWorker(logger *log.Entry, msgHandler MessageHandler, workerCh <-chan *corepb.EdgeCPMessage) {
	go func() {
		for {
			select {
			case <-p.done:
				return
			case e2CpMsg := <-workerCh:
				msgType, err := types.AnyMessageName(e2CpMsg.Msg)
				if err != nil {
					logger.WithError(err).WithField("deviceID", e2CpMsg.Metadata.DeviceID).Error("Failed to get type of any message")
				}

				msgHandler.HandleMessage(e2CpMsg.Metadata, e2CpMsg.Msg, msgType)
			}
		}
	}()
}

// Stop performs any necessary cleanup before shutdown.
func (p *PartitionHandler) Stop() {
	p.once.Do(func() {
		close(p.done)
	})
}
