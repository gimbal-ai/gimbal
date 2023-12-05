/*
 * Copyright 2018- The Pixie Authors.
 * Modifications Copyright 2023- Gimlet Labs, Inc.
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

package msgbus_test

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"testing"
	"time"

	"github.com/nats-io/nats.go/jetstream"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/shared/services/msgbus"
	"gimletlabs.ai/gimlet/src/shared/services/natstest"
)

var testStreamCfg = jetstream.StreamConfig{
	Name:     "abc",
	Subjects: []string{"abc"},
	MaxAge:   2 * time.Minute,
	Replicas: 1,
	Storage:  jetstream.MemoryStorage,
}

func receiveExpectedUpdates(c <-chan jetstream.Msg, data [][]byte) error {
	// On some investigations, found that messages are sent < 500us, chose a
	// timeout that was 100x in case of any unexpected interruptions.
	timeout := 50 * time.Millisecond

	// We do not early return any errors, otherwise we won't consume all messages
	// sent and risk race conditions in tests.
	// If no errors are reached, `err` will be nil.
	var err error

	curr := 0
	for {
		select {
		case m := <-c:
			if curr >= len(data) {
				err = fmt.Errorf("unexpected message: %s", string(m.Data()))
			} else if !bytes.Equal(data[curr], m.Data()) {
				err = fmt.Errorf("data doesn't match on update %d", curr)
			}
			curr++
		case <-time.After(timeout):
			if curr < len(data) {
				return errors.New("timed out waiting for messages on subscription")
			}
			return err
		}
	}
}

func TestJetStream_PersistentSubscribeInterfaceAccuracy(t *testing.T) {
	sub := testStreamCfg.Name
	data := [][]byte{[]byte("123"), []byte("abc"), []byte("asdf")}

	nc := natstest.MustStartTestNATS(t)

	js := msgbus.MustConnectJetStream(nc)

	_, err := js.CreateStream(context.Background(), testStreamCfg)
	require.NoError(t, err)

	s, err := msgbus.NewJetStreamStreamer(js, sub)
	require.NoError(t, err)

	// Publish data to the subject.
	for _, d := range data {
		require.NoError(t, s.Publish(sub, d))
	}

	ctx := context.Background()
	_, err = js.CreateOrUpdateConsumer(ctx, sub, jetstream.ConsumerConfig{
		Durable:       "abc|indexer",
		FilterSubject: sub,
		DeliverPolicy: jetstream.DeliverAllPolicy,
		AckWait:       30 * time.Second,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxAckPending: 50,
	})
	require.NoError(t, err)
	ch1 := make(chan jetstream.Msg)
	pSub, err := s.PersistentSubscribe(sub, "indexer", func(m jetstream.Msg) {
		ch1 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)

	// Should receive all messages that were published.
	require.NoError(t, receiveExpectedUpdates(ch1, data))
	pSub.Close()

	// Make sure when we recreate the subscription, we don't receive new messages (all old ack messages should be ignored).
	ch2 := make(chan jetstream.Msg)
	pSub, err = s.PersistentSubscribe(sub, "indexer", func(m jetstream.Msg) {
		ch2 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)

	// Should receive no messages.
	require.NoError(t, receiveExpectedUpdates(ch2, [][]byte{}))
	pSub.Close()

	// New durable subscribe with a different name should receive all of the old updates.
	ctx = context.Background()
	_, err = js.CreateOrUpdateConsumer(ctx, sub, jetstream.ConsumerConfig{
		Durable:       "abc|new_indexer",
		FilterSubject: sub,
		DeliverPolicy: jetstream.DeliverAllPolicy,
		AckWait:       30 * time.Second,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxAckPending: 50,
	})
	require.NoError(t, err)
	ch3 := make(chan jetstream.Msg)
	pSub, err = s.PersistentSubscribe(sub, "new_indexer", func(m jetstream.Msg) {
		ch3 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)

	// Should receive all messages on this channel.
	require.NoError(t, receiveExpectedUpdates(ch3, data))
	pSub.Close()
}

func TestJetStream_PersistentSubscribeMultiConsumer(t *testing.T) {
	sub := testStreamCfg.Name
	data := [][]byte{[]byte("123"), []byte("abc"), []byte("asdf")}

	nc := natstest.MustStartTestNATS(t)
	js := msgbus.MustConnectJetStream(nc)

	_, err := js.CreateStream(context.Background(), testStreamCfg)
	require.NoError(t, err)

	s, err := msgbus.NewJetStreamStreamer(js, sub)
	require.NoError(t, err)

	// Publish data to the subject.
	for _, d := range data {
		require.NoError(t, s.Publish(sub, d))
	}

	ctx := context.Background()
	_, err = js.CreateOrUpdateConsumer(ctx, sub, jetstream.ConsumerConfig{
		Durable:       "abc|indexer",
		FilterSubject: sub,
		DeliverPolicy: jetstream.DeliverAllPolicy,
		AckWait:       30 * time.Second,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxAckPending: 50,
	})
	require.NoError(t, err)

	ch1 := make(chan jetstream.Msg)
	pSub1, err := s.PersistentSubscribe(sub, "indexer", func(m jetstream.Msg) {
		ch1 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)

	ch2 := make(chan jetstream.Msg)
	pSub2, err := s.PersistentSubscribe(sub, "indexer", func(m jetstream.Msg) {
		ch2 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)

	var out [][]byte
	func() {
		for {
			select {
			case m := <-ch1:
				out = append(out, m.Data())
			case m := <-ch2:
				out = append(out, m.Data())
			case <-time.After(500 * time.Millisecond):
				return
			}
		}
	}()

	pSub1.Close()
	pSub2.Close()
	assert.ElementsMatch(t, data, out)

	ch3 := make(chan jetstream.Msg)
	pSub3, err := s.PersistentSubscribe(sub, "indexer", func(m jetstream.Msg) {
		ch3 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)
	require.NoError(t, s.Publish(sub, []byte("new_data_1")))
	require.NoError(t, s.Publish(sub, []byte("new_data_2")))
	// Should receive only new messages.
	require.NoError(t, receiveExpectedUpdates(ch3, [][]byte{[]byte("new_data_1"), []byte("new_data_2")}))
	pSub3.Close()
}

func TestJetStream_PublishAfterSubscribe(t *testing.T) {
	sub := testStreamCfg.Name
	data := [][]byte{[]byte("123"), []byte("abc"), []byte("asdf")}

	nc := natstest.MustStartTestNATS(t)
	js := msgbus.MustConnectJetStream(nc)

	_, err := js.CreateStream(context.Background(), testStreamCfg)
	require.NoError(t, err)

	s, err := msgbus.NewJetStreamStreamer(js, sub)
	require.NoError(t, err)

	ctx := context.Background()
	_, err = js.CreateOrUpdateConsumer(ctx, sub, jetstream.ConsumerConfig{
		Durable:       "abc|indexer",
		FilterSubject: sub,
		DeliverPolicy: jetstream.DeliverAllPolicy,
		AckWait:       30 * time.Second,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxAckPending: 50,
	})
	require.NoError(t, err)

	// Subscribe first to the data.
	ch1 := make(chan jetstream.Msg)
	pSub, err := s.PersistentSubscribe(sub, "indexer", func(m jetstream.Msg) {
		ch1 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)

	// Then publish data to the subject.
	for _, d := range data {
		require.NoError(t, s.Publish(sub, d))
	}

	// Should receive all messages that were published.
	require.NoError(t, receiveExpectedUpdates(ch1, data))
	pSub.Close()
}

func TestJetStream_PersistentSubscribeReattemptAck(t *testing.T) {
	sub := testStreamCfg.Name
	data := [][]byte{[]byte("123"), []byte("abc"), []byte("asdf")}

	// Test to make sure that not-acking a message will make sure that it comes back.
	nc := natstest.MustStartTestNATS(t)
	js := msgbus.MustConnectJetStream(nc)

	_, err := js.CreateStream(context.Background(), testStreamCfg)
	require.NoError(t, err)

	s, err := msgbus.NewJetStreamStreamer(js, sub)
	require.NoError(t, err)

	// Publish data to the subject.
	for _, d := range data {
		require.NoError(t, s.Publish(sub, d))
	}

	ctx := context.Background()
	_, err = js.CreateOrUpdateConsumer(ctx, sub, jetstream.ConsumerConfig{
		Durable:       "abc|indexer",
		FilterSubject: sub,
		DeliverPolicy: jetstream.DeliverAllPolicy,
		AckWait:       100 * time.Millisecond,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxAckPending: 50,
	})
	require.NoError(t, err)

	ch4 := make(chan jetstream.Msg)
	first := true
	pSub, err := s.PersistentSubscribe(sub, "indexer", func(m jetstream.Msg) {
		if first {
			first = false
			return
		}
		ch4 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)

	// Receive all but the first data point.
	require.NoError(t, receiveExpectedUpdates(ch4, data[1:]))

	time.Sleep(100 * time.Millisecond)

	// Receive the last missing datapoint.
	require.NoError(t, receiveExpectedUpdates(ch4, data[0:1]))
	pSub.Close()
}

func TestJetStream_MultiSubjectStream(t *testing.T) {
	sub := "abc"
	nc := natstest.MustStartTestNATS(t)
	js := msgbus.MustConnectJetStream(nc)

	_, err := js.CreateStream(context.Background(), jetstream.StreamConfig{
		Name:     sub,
		Subjects: []string{"abc", "abc.*", "abc.*.*", "abc.*.*.*"},
		MaxAge:   time.Minute * 2,
	})
	require.NoError(t, err)

	s, err := msgbus.NewJetStreamStreamer(js, sub)
	require.NoError(t, err)

	// Should be able to publish and receive single nested.
	ctx := context.Background()
	_, err = js.CreateOrUpdateConsumer(ctx, sub, jetstream.ConsumerConfig{
		Durable:       "abc__|indexer",
		FilterSubject: "abc.*",
		DeliverPolicy: jetstream.DeliverAllPolicy,
		AckWait:       30 * time.Second,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxAckPending: 50,
	})
	require.NoError(t, err)
	ch1 := make(chan jetstream.Msg)
	pSub, err := s.PersistentSubscribe("abc.*", "indexer", func(m jetstream.Msg) {
		ch1 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)
	require.NoError(t, s.Publish("abc.blah", []byte("abc")))
	require.NoError(t, receiveExpectedUpdates(ch1, [][]byte{[]byte("abc")}))
	pSub.Close()

	// Should be able to publish and receive double nested.
	ctx = context.Background()
	_, err = js.CreateOrUpdateConsumer(ctx, sub, jetstream.ConsumerConfig{
		Durable:       "abc____|indexer",
		FilterSubject: "abc.*.*",
		DeliverPolicy: jetstream.DeliverAllPolicy,
		AckWait:       30 * time.Second,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxAckPending: 50,
	})
	require.NoError(t, err)
	ch2 := make(chan jetstream.Msg)
	pSub, err = s.PersistentSubscribe("abc.*.*", "indexer", func(m jetstream.Msg) {
		ch2 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)
	require.NoError(t, s.Publish("abc.blah.blah", []byte("asdf")))
	require.NoError(t, receiveExpectedUpdates(ch2, [][]byte{[]byte("asdf")}))
	pSub.Close()

	// Should be able to publish and receive triple nested.
	ch3 := make(chan jetstream.Msg)
	ctx = context.Background()
	_, err = js.CreateOrUpdateConsumer(ctx, sub, jetstream.ConsumerConfig{
		Durable:       "abc______|indexer",
		FilterSubject: "abc.*.*.*",
		DeliverPolicy: jetstream.DeliverAllPolicy,
		AckWait:       30 * time.Second,
		AckPolicy:     jetstream.AckExplicitPolicy,
		MaxAckPending: 50,
	})
	require.NoError(t, err)
	pSub, err = s.PersistentSubscribe("abc.*.*.*", "indexer", func(m jetstream.Msg) {
		ch3 <- m
		require.NoError(t, m.Ack())
	})
	require.NoError(t, err)
	require.NoError(t, s.Publish("abc.blah.blah.blah", []byte("bteg")))
	require.NoError(t, receiveExpectedUpdates(ch3, [][]byte{[]byte("bteg")}))
	pSub.Close()
}
