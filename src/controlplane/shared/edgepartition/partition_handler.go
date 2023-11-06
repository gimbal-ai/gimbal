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

type MessageHandler func(*corepb.EdgeCPMetadata, *types.Any)

// PartitionHandler handles any incoming NATS messages from an edge device.
type PartitionHandler struct {
	nc              *nats.Conn
	e2cpTopic       corepb.EdgeCPTopic
	messageHandlers map[string]MessageHandler
	msgCh           chan *corepb.EdgeCPMessage

	// Signal used to quit.
	done chan struct{}
	once sync.Once
}

// NewPartitionHandler creates a new partition handler for the given topic.
func NewPartitionHandler(nc *nats.Conn, e2cpTopic corepb.EdgeCPTopic, messageHandlers map[string]MessageHandler) *PartitionHandler {
	return &PartitionHandler{nc: nc, e2cpTopic: e2cpTopic, messageHandlers: messageHandlers, msgCh: make(chan *corepb.EdgeCPMessage, channelSize)}
}

// Start starts the listening and handling for messages from any edge devices.
func (p *PartitionHandler) Start() error {
	// Subscribe to NATS channel for partition.
	log.Info("Subscribing to NATS channels")
	partitions := GenerateRange()
	for _, pr := range partitions {
		subCh := make(chan *nats.Msg, channelSize)
		err := p.startPartitionHandler(pr, subCh)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *PartitionHandler) startPartitionHandler(partition string, subCh chan *nats.Msg) error {
	topic, err := EdgeToCPNATSPartitionTopic(partition, p.e2cpTopic, false)
	if err != nil {
		return err
	}
	topicLog := log.WithField("topic", topic)

	natsSub, err := p.nc.ChanSubscribe(topic, subCh)
	if err != nil {
		topicLog.WithError(err).Error("Failed to subscribe to NATS")
		return err
	}

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
				p.msgCh <- e2CpMsg
			}
		}
	}()

	go func() {
		for {
			select {
			case <-p.done:
				return
			case e2CpMsg := <-p.msgCh:
				msgType, err := types.AnyMessageName(e2CpMsg.Msg)
				if err != nil {
					topicLog.WithError(err).WithField("deviceID", e2CpMsg.Metadata.DeviceID).Error("Failed to get type of any message")
				}

				funcHandler, ok := p.messageHandlers[msgType]
				if !ok {
					topicLog.WithField("msgType", msgType).WithField("deviceID", e2CpMsg.Metadata.DeviceID).Error("Message type does not match any expected messages")
					continue
				}

				funcHandler(e2CpMsg.Metadata, e2CpMsg.Msg)
			}
		}
	}()

	return nil
}

// Stop performs any necessary cleanup before shutdown.
func (p *PartitionHandler) Stop() {
	p.once.Do(func() {
		close(p.done)
	})
}
