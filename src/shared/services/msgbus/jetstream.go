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

package msgbus

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/nats-io/nats.go"
	"github.com/nats-io/nats.go/jetstream"
	log "github.com/sirupsen/logrus"
)

const (
	publishRetryInterval = 25 * time.Millisecond
	publishTimeout       = 1 * time.Minute
	fetchTimeout         = 5 * time.Second
)

// MustConnectJetStream creates a new JetStream connection.
func MustConnectJetStream(nc *nats.Conn) jetstream.JetStream {
	js, err := jetstream.New(nc)
	if err != nil {
		log.WithError(err).Fatal("Could not connect to Jetstream")
	}

	return js
}

// JetStreamStreamer is a Streamer implemented using JetStream.
type JetStreamStreamer struct {
	js     jetstream.JetStream
	stream jetstream.Stream

	bOpts *backoff.ExponentialBackOff
}

// NewJetStreamStreamer creates a new Streamer implemented using JetStream.
func NewJetStreamStreamer(js jetstream.JetStream, streamName string) (*JetStreamStreamer, error) {
	ctx, ctxCancel := context.WithTimeout(context.Background(), fetchTimeout)
	defer ctxCancel()
	stream, err := js.Stream(ctx, streamName)
	if err != nil {
		return nil, err
	}

	bOpts := backoff.NewExponentialBackOff()
	bOpts.InitialInterval = publishRetryInterval
	bOpts.MaxElapsedTime = publishTimeout

	return &JetStreamStreamer{
		js:     js,
		stream: stream,
		bOpts:  bOpts,
	}, nil
}

func (s *JetStreamStreamer) Publish(subject string, data []byte) error {
	return backoff.Retry(func() error {
		pubFuture, err := s.js.PublishAsync(subject, data)
		if err != nil {
			log.WithError(err).Error("failed to publish")
			return err
		}
		select {
		case <-pubFuture.Ok():
			return nil
		case err = <-pubFuture.Err():
			log.WithError(err).Error("failed to publish2")
			return err
		}
	}, s.bOpts)
}

// MsgHandler is a function that processes Msg.
type MsgHandler func(msg jetstream.Msg)

// PersistentSub is the interface to an active persistent subscription.
type PersistentSub interface {
	// Close the subscription, but allow future PersistentSubs to read from the sub starting after
	// the last acked message.
	Close()
}

// persistentJetStreamSub is a wrapper around the JetStream subscription that implements the PersistentSub interface.
type persistentJetStreamSub struct {
	cons jetstream.ConsumeContext
}

func (s *persistentJetStreamSub) Close() {
	s.cons.Stop()
}

func (s *JetStreamStreamer) PersistentSubscribe(subject, persistentName string, cb MsgHandler) (PersistentSub, error) {
	consumerName := FormatConsumerName(subject, persistentName)

	ctx, ctxCancel := context.WithTimeout(context.Background(), fetchTimeout)
	defer ctxCancel()
	streamInfo, err := s.stream.Info(ctx)
	if err != nil {
		return nil, err
	}

	consumer, err := s.js.Consumer(ctx, streamInfo.Config.Name, consumerName)
	if err != nil {
		return nil, err
	}

	cons, err := consumer.Consume(func(msg jetstream.Msg) {
		cb(msg)
	})
	if err != nil {
		return nil, err
	}

	return &persistentJetStreamSub{cons}, nil
}

// FormatConsumerName returns the jetstream consumer name given a subject and persistent name.
func FormatConsumerName(subject, persistentName string) string {
	return fmt.Sprintf("%s|%s", strings.ReplaceAll(strings.ReplaceAll(subject, ".", "_"), "*", "_"), persistentName)
}

func GetPersistentNameFromConsumerName(consumerName string) (string, error) {
	splitName := strings.Split(consumerName, "|")
	if len(splitName) < 2 {
		return "", errors.New("invalid consumerName")
	}
	return splitName[1], nil
}
