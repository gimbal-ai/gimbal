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

package victoriametrics

import (
	"github.com/prometheus/client_golang/api"
	v1 "github.com/prometheus/client_golang/api/prometheus/v1"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

func init() {
	pflag.String("victoriametrics_address", "http://localhost:8428", "The hostname for victoriametrics")
}

func MustConnectVictoriaMetrics() v1.API {
	client, err := api.NewClient(api.Config{
		Address: viper.GetString("victoriametrics_address"),
	})
	if err != nil {
		log.WithError(err).Fatalf("failed to connect for victoriametrics")
	}

	return v1.NewAPI(client)
}
