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

package events

// This package provides a small wrapper around segment to provide initialization and a dummy
// writer for development mode. The dummy writer is used if segment credentials are not set.

import (
	"sync"

	"github.com/segmentio/analytics-go/v3"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

func init() {
	pflag.String("segment_write_key", "", "The key to use for segment")
	pflag.Int("segment_batch_size", analytics.DefaultBatchSize, "The batch size to use for segment")
	pflag.Duration("segment_flush_interval", analytics.DefaultInterval, "The flush interval for segment")
}

var (
	client analytics.Client
	once   sync.Once
)

type placeholderClient struct{}

func (*placeholderClient) Enqueue(msg analytics.Message) error {
	if err := msg.Validate(); err != nil {
		return err
	}

	log.WithField("msg", msg).Debug("Placeholder analytics client, dropping message...")
	return nil
}

func (*placeholderClient) Close() error {
	return nil
}

func getDefaultClient() analytics.Client {
	k := viper.GetString("segment_write_key")
	// Key is specified try to to create segment client.
	if len(k) > 0 {
		c, err := analytics.NewWithConfig(k, analytics.Config{
			BatchSize: viper.GetInt("segment_batch_size"),
			Interval:  viper.GetDuration("segment_flush_interval"),
		})
		if err != nil {
			log.WithError(err).Error("Failed to create segment client")
			return &placeholderClient{}
		}
		return c
	}
	return &placeholderClient{}
}

// SetClient sets the default client used for event tracking.
func SetClient(c analytics.Client) {
	client = c
}

// Client returns the client.
func Client() analytics.Client {
	once.Do(func() {
		// client has already been set up.
		if client != nil {
			return
		}
		SetClient(getDefaultClient())
	})
	return client
}
