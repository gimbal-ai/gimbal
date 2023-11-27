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
	"errors"
	"sync"

	"github.com/gogo/protobuf/types"
	"github.com/nats-io/nats.go/jetstream"
	log "github.com/sirupsen/logrus"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/shared/services/msgbus"
)

var ErrInvalidMessage = errors.New("message format invalid or unexpected")

type DurableMessageHandler func(*corepb.EdgeCPMetadata, *types.Any) error

type partitionTopic func(partition string) (string, error)

// DurablePartitionHandler handles any incoming Jetstream messages from an edge device.
type DurablePartitionHandler struct {
	js              *msgbus.JetStreamStreamer
	partitionTopic  partitionTopic
	consumer        string
	messageHandlers map[string]DurableMessageHandler

	// Signal used to quit.
	done chan struct{}
	once sync.Once
}

// NewDurablePartitionHandler creates a new partition handler for the given topic.
func NewDurablePartitionHandler(js *msgbus.JetStreamStreamer, consumer string, messageHandlers map[string]DurableMessageHandler) *DurablePartitionHandler {
	return &DurablePartitionHandler{
		js:              js,
		consumer:        consumer,
		messageHandlers: messageHandlers,
	}
}

// Start starts the listening and handling for messages from any edge devices.
func (p *DurablePartitionHandler) Start() error {
	// Subscribe to NATS channel for partition.
	log.Info("Subscribing to JetStream channels")
	partitions := GenerateRange()
	for _, pr := range partitions {
		err := p.startDurablePartitionHandler(pr)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *DurablePartitionHandler) WithEdgeToCPTopic(topic corepb.EdgeCPTopic) *DurablePartitionHandler {
	p.partitionTopic = func(p string) (string, error) {
		return EdgeToCPNATSPartitionTopic(p, topic, true)
	}

	return p
}

func (p *DurablePartitionHandler) WithCPTopic(topic corepb.CPTopic) *DurablePartitionHandler {
	p.partitionTopic = func(p string) (string, error) {
		return CPNATSPartitionTopic(p, topic, true)
	}

	return p
}

func (p *DurablePartitionHandler) handleMessage(topic string, msg jetstream.Msg) error {
	topicLog := log.WithField("topic", topic)

	e2CpMsg := &corepb.EdgeCPMessage{}
	err := e2CpMsg.Unmarshal(msg.Data())
	if err != nil {
		return err
	}

	msgType, err := types.AnyMessageName(e2CpMsg.Msg)
	if err != nil {
		topicLog.WithError(err).WithField("deviceID", e2CpMsg.Metadata.DeviceID).Error("Failed to get type of any message")
		return ErrInvalidMessage
	}

	funcHandler, ok := p.messageHandlers[msgType]
	if !ok {
		topicLog.WithField("msgType", msgType).WithField("deviceID", e2CpMsg.Metadata.DeviceID).Error("Message type does not match any expected messages")
		return ErrInvalidMessage
	}

	return funcHandler(e2CpMsg.Metadata, e2CpMsg.Msg)
}

func (p *DurablePartitionHandler) startDurablePartitionHandler(partition string) error {
	topic, err := p.partitionTopic(partition)
	if err != nil {
		return err
	}

	sub, err := p.js.PersistentSubscribe(topic, p.consumer, func(msg jetstream.Msg) {
		err = p.handleMessage(topic, msg)
		if err != nil {
			err = msg.Nak()
			if err != nil && !errors.Is(err, jetstream.ErrMsgAlreadyAckd) {
				log.WithError(err).Fatal("Failed to nack Jetstream message")
			}
			return
		}

		err = msg.Ack()
		if err != nil && !errors.Is(err, jetstream.ErrMsgAlreadyAckd) {
			log.WithError(err).Fatal("Failed to ack Jetstream message")
		}
	})
	if err != nil {
		return err
	}

	go func() {
		<-p.done
		sub.Close()
	}()

	return nil
}

// Stop performs any necessary cleanup before shutdown.
func (p *DurablePartitionHandler) Stop() {
	p.once.Do(func() {
		close(p.done)
	})
}
