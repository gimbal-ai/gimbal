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
