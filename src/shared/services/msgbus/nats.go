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

package msgbus

import (
	"github.com/nats-io/nats.go"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

func init() {
	pflag.String("nats_url", "gml-nats", "The url of the nats message bus")
}

// MustConnectNATS attempts to connect to the NATS message bus.
func MustConnectNATS() *nats.Conn {
	var nc *nats.Conn
	var err error
	natsURL := viper.GetString("nats_url")
	if viper.GetBool("disable_ssl") {
		nc, err = nats.Connect(natsURL)
	} else {
		nc, err = nats.Connect(natsURL,
			nats.ClientCert(viper.GetString("client_tls_cert"), viper.GetString("client_tls_key")),
			nats.RootCAs(viper.GetString("tls_ca_cert")))
	}

	if err != nil && !viper.GetBool("disable_ssl") {
		log.WithError(err).
			WithField("nats_url", natsURL).
			WithField("client_tls_cert", viper.GetString("client_tls_cert")).
			WithField("client_tls_key", viper.GetString("client_tls_key")).
			WithField("tls_ca_cert", viper.GetString("tls_ca_cert")).
			Fatal("Failed to connect to NATS")
	} else if err != nil {
		log.WithError(err).WithField("nats_url", natsURL).Fatal("Failed to connect to NATS")
	}
	nc.SetErrorHandler(func(_ *nats.Conn, subscription *nats.Subscription, err error) {
		log.WithField("Sub", subscription.Subject).
			WithError(err).
			Error("Error with NATS handler")
	})
	log.WithField("URL", natsURL).Info("Connected to NATS")
	return nc
}
