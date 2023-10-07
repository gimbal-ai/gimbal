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

package main

import (
	"time"

	log "github.com/sirupsen/logrus"

	"gimletlabs.ai/gimlet/src/controlplane/shared/idempotency_cleanup/utils"
	"gimletlabs.ai/gimlet/src/shared/services/pg"
	"gimletlabs.ai/gimlet/src/shared/services/server"
)

const (
	// Expire out idempotency keys every 24 hours.
	expiryDuration = 24 * time.Hour
)

func main() {
	server.PostFlagSetupAndParse()
	server.CheckServiceFlags()

	db := pg.MustConnectDefaultPostgresDB()
	err := utils.ExpireKeys(db, expiryDuration)
	if err != nil {
		log.WithError(err).Fatal("Failed to expire keys")
	}
}
