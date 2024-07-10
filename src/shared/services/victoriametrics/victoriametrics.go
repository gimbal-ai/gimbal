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

package victoriametrics

import (
	"net"
	"net/http"
	"net/url"
	"strings"

	"github.com/prometheus/client_golang/api"
	promv1 "github.com/prometheus/client_golang/api/prometheus/v1"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
)

var client = &http.Client{}

func init() {
	pflag.String("victoriametrics_insert_scheme", "http", "The scheme for victoriametrics insert service")
	pflag.String("victoriametrics_insert_host", "localhost", "The hostname for victoriametrics insert service")
	pflag.String("victoriametrics_insert_port", "8480", "The port for victoriametrics insert service")

	pflag.String("victoriametrics_select_scheme", "http", "The scheme for victoriametrics select service")
	pflag.String("victoriametrics_select_host", "localhost", "The hostname for victoriametrics select service")
	pflag.String("victoriametrics_select_port", "8481", "The port for victoriametrics select service")
	pflag.String("victoriametrics_select_path_prefix", "", "The path prefix for the victoriametrics select service, needed for clustered mode")
}

func GetVictoriaMetricsURL() *url.URL {
	u := &url.URL{
		Scheme: viper.GetString("victoriametrics_select_scheme"),
		Host:   net.JoinHostPort(viper.GetString("victoriametrics_select_host"), viper.GetString("victoriametrics_select_port")),
		Path:   viper.GetString("victoriametrics_select_path_prefix"),
	}
	return u
}

func MustConnectVictoriaMetricsSelect() promv1.API {
	client, err := api.NewClient(
		api.Config{
			Address: GetVictoriaMetricsURL().String(),
		})
	if err != nil {
		log.WithError(err).Fatalf("failed to connect for victoriametrics select")
	}

	return promv1.NewAPI(client)
}

func InsertPrometheusMetrics(data string) error {
	u := url.URL{
		Scheme: viper.GetString("victoriametrics_insert_scheme"),
		Host:   net.JoinHostPort(viper.GetString("victoriametrics_insert_host"), viper.GetString("victoriametrics_insert_port")),
		Path:   "/api/v1/import/prometheus",
	}

	req, err := http.NewRequest("POST", u.String(), strings.NewReader(data))
	if err != nil {
		return err
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	return resp.Body.Close()
}
