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

package natstest

import (
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/cenkalti/backoff/v4"
	"github.com/nats-io/nats-server/v2/server"
	"github.com/nats-io/nats-server/v2/test"
	"github.com/nats-io/nats.go"
	"github.com/phayes/freeport"
)

func startNATS() (*server.Server, *nats.Conn, error) {
	var err error
	defer func() {
		if r := recover(); r != nil {
			err = errors.New("Could not run NATS server")
		}
	}()
	// Find available port.
	port, err := freeport.GetFreePort()
	if err != nil {
		return nil, nil, err
	}

	opts := test.DefaultTestOptions
	opts.ServerName = "test_nats_js"
	opts.JetStream = true
	opts.Port = port
	dir, err := os.MkdirTemp("", "nats")
	if err != nil {
		return nil, nil, err
	}
	opts.StoreDir = dir

	gnatsd := test.RunServer(&opts)
	if gnatsd == nil {
		return nil, nil, errors.New("Could not run NATS server")
	}

	url := fmt.Sprintf("nats://%s:%d", opts.Host, opts.Port)
	conn, err := nats.Connect(url)
	if err != nil {
		gnatsd.Shutdown()
		return nil, nil, err
	}
	return gnatsd, conn, err
}

// MustStartTestNATS starts up a NATS server at an open port.
func MustStartTestNATS(t *testing.T) (*nats.Conn, func()) {
	var gnatsd *server.Server
	var conn *nats.Conn

	natsConnectFn := func() error {
		var err error
		gnatsd, conn, err = startNATS()
		if gnatsd == nil && conn == nil { // Handle case where startNATS has a recover.
			err = errors.New("Failed to connect to NATS")
		}
		if err != nil {
			return err
		}
		return nil
	}

	bo := backoff.NewExponentialBackOff()
	bo.MaxInterval = 5 * time.Second
	bo.MaxElapsedTime = 1 * time.Minute

	err := backoff.Retry(natsConnectFn, bo)
	if err != nil {
		t.Fatal("Could not connect to NATS")
	}

	cleanup := func() {
		dir := gnatsd.StoreDir()
		gnatsd.Shutdown()
		os.RemoveAll(dir)
		conn.Close()
	}

	return conn, cleanup
}
