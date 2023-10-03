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

package server

import (
	"os"
	"runtime"
	"time"

	"github.com/getsentry/sentry-go"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"

	version "gimletlabs.ai/gimlet/src/shared/goversion"
	"gimletlabs.ai/gimlet/src/shared/services/sentryhook"
)

func init() {
	pflag.String("sentry_dsn", "", "The sentry DSN. Empty disables sentry")
}

// InitDefaultSentry initialize sentry with default options. The options are set based on values
// from viper.
func InitDefaultSentry() func() {
	dsn := viper.GetString("sentry_dsn")

	return InitSentryWithDSN("", dsn)
}

// InitSentryWithDSN initializes sentry with the given DSN.
func InitSentryWithDSN(id, dsn string) func() {
	podName := viper.GetString("pod_name")
	executable, _ := os.Executable()

	err := sentry.Init(sentry.ClientOptions{
		Dsn:              dsn,
		AttachStacktrace: true,
		Release:          version.GetVersion().ToString(),
		Environment:      runtime.GOOS,
		MaxBreadcrumbs:   10,
	})
	if err != nil {
		log.WithError(err).Trace("Cannot initialize sentry")
	} else {
		tags := map[string]string{
			"version":    version.GetVersion().ToString(),
			"executable": executable,
			"pod_name":   podName,
		}
		if id != "" {
			tags["ID"] = id
		}
		hook := sentryhook.New([]log.Level{
			log.ErrorLevel, log.PanicLevel, log.FatalLevel,
		}, sentryhook.WithTags(tags))
		log.AddHook(hook)
	}

	return func() {
		sentry.Flush(2 * time.Second)
	}
}
