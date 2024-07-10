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

package edgepartition

import (
	"context"
	"errors"
	"fmt"
	"sync"

	"github.com/gogo/protobuf/types"
	"github.com/nats-io/nats.go/jetstream"
	log "github.com/sirupsen/logrus"

	"gimletlabs.ai/gimlet/src/api/corepb/v1"
	"gimletlabs.ai/gimlet/src/common/typespb"
	"gimletlabs.ai/gimlet/src/shared/services/msgbus"
)

var ErrInvalidMessage = errors.New("message format invalid or unexpected")

type MsgMetadata struct {
	*corepb.EdgeCPMetadata
	*corepb.CPMetadata
	Msg jetstream.Msg
}

type DurableMessageHandler func(context.Context, *MsgMetadata, *types.Any) error

type partitionTopic func(partition string) (string, error)

// DurablePartitionHandler handles any incoming Jetstream messages from an edge device.
type DurablePartitionHandler struct {
	js              *msgbus.JetStreamStreamer
	partitionTopic  partitionTopic
	consumer        string
	messageHandlers map[string]DurableMessageHandler
	isCPTopic       bool
	ctx             context.Context
	ctxCancel       func()

	// Signal used to quit.
	done chan struct{}
	once sync.Once
}

// NewDurablePartitionHandler creates a new partition handler for the given topic.
func NewDurablePartitionHandler(js *msgbus.JetStreamStreamer, consumer string, messageHandlers map[string]DurableMessageHandler) *DurablePartitionHandler {
	ctx, cancel := context.WithCancel(context.Background())
	return &DurablePartitionHandler{
		js:              js,
		consumer:        consumer,
		messageHandlers: messageHandlers,
		ctx:             ctx,
		ctxCancel:       cancel,
		done:            make(chan struct{}),
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
	log.Info("Finished subscribing to JetStream channels")
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
	p.isCPTopic = true

	return p
}

func (p *DurablePartitionHandler) handleMessage(topic string, msg jetstream.Msg) error {
	topicLog := log.WithField("topic", topic)

	var anyMsg *types.Any
	var entityID *typespb.UUID
	md := &MsgMetadata{}
	md.Msg = msg

	if p.isCPTopic {
		cpMsg := &corepb.CPMessage{}
		err := cpMsg.Unmarshal(msg.Data())
		if err != nil {
			return err
		}
		anyMsg = cpMsg.Msg
		entityID = cpMsg.Metadata.EntityID
		md.CPMetadata = cpMsg.Metadata
	} else {
		e2CpMsg := &corepb.EdgeCPMessage{}
		err := e2CpMsg.Unmarshal(msg.Data())
		if err != nil {
			return err
		}
		anyMsg = e2CpMsg.Msg
		entityID = e2CpMsg.Metadata.DeviceID
		md.EdgeCPMetadata = e2CpMsg.Metadata
	}

	msgType, err := types.AnyMessageName(anyMsg)
	if err != nil {
		topicLog.WithError(err).WithField("entityID", entityID).Error("Failed to get type of any message")
		return ErrInvalidMessage
	}

	funcHandler, ok := p.messageHandlers[msgType]
	if !ok {
		topicLog.WithField("msgType", msgType).WithField("entityID", entityID).Error("Message type does not match any expected messages")
		return ErrInvalidMessage
	}

	return funcHandler(p.ctx, md, anyMsg)
}

func (p *DurablePartitionHandler) startDurablePartitionHandler(partition string) error {
	topic, err := p.partitionTopic(partition)
	if err != nil {
		return err
	}

	sub, err := p.js.PersistentSubscribe(topic, p.consumer, func(msg jetstream.Msg) {
		err = p.handleMessage(topic, msg)
		// We still ack the message even if there is an error handling it, that way we don't block up the consumer queue.
		if err != nil {
			// TODO(philkuz,GML-286): Create a sentry message when this error occurs.
			consumerSequence := "not retrieved"
			streamSequence := "not retrieved"
			if metadata, err := msg.Metadata(); err == nil {
				consumerSequence = fmt.Sprintf("%d", metadata.Sequence.Consumer)
				streamSequence = fmt.Sprintf("%d", metadata.Sequence.Stream)
			} else {
				log.WithError(err).Error("Failed to retrieve Jetstream message metadata during error handling")
			}
			log.WithError(err).WithFields(log.Fields{
				"consumerSeq": consumerSequence,
				"streamSeq":   streamSequence,
				"subject":     msg.Subject(),
				"topic":       topic,
			}).Error("Failed to handle Jetstream message. Dropping message")
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
		p.ctxCancel()
		close(p.done)
	})
}
