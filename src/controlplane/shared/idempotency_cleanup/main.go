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

package main

import (
	"context"
	"net/http"
	"time"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"

	"gimletlabs.ai/gimlet/src/controlplane/shared/idempotency_cleanup/utils"
	"gimletlabs.ai/gimlet/src/shared/services/pg"
	"gimletlabs.ai/gimlet/src/shared/services/server"
)

const (
	// Expire out idempotency keys every 24 hours.
	expiryDuration = 24 * time.Hour
)

func init() {
	pflag.String("db_proxy_stop_path", "", "The path to send a POST request to trigger the proxy to terminate")
}

func main() {
	server.PostFlagSetupAndParse()

	db := pg.MustConnectDefaultPostgresDB()
	err := utils.ExpireKeys(context.Background(), db, expiryDuration)
	if err != nil {
		log.WithError(err).Fatal("Failed to expire keys")
	}

	dbStopPath := viper.GetString("db_proxy_stop_path")
	// Trigger stop of DB.
	if dbStopPath != "" {
		_, err = http.Post(dbStopPath, "application/json", nil)
		if err != nil {
			log.WithError(err).Error("Failed to call quit")
		}
	}
}
