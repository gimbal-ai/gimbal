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

package statusz

import (
	"net/http"

	log "github.com/sirupsen/logrus"
)

// statusCheckFn is the function that is called to determine the status for the statusz endpoint.
// A non-empty returned string is considered a non-ready status and considered a 503 ServiceUnavailable.
// An empty string is considered to be a 200 OK.
type statusCheckFn func() string

// mux is an interface describing the methods InstallHandler requires.
type mux interface {
	Handle(pattern string, handler http.Handler)
}

// InstallPathHandler registers the statusz checks under path.
// This function can only be called once per mux/path combo.
// Our use of the statusz endpoint is to return a simple human-readable string
// containing the reason for why the pod is in its current state.
func InstallPathHandler(mux mux, path string, statusChecker statusCheckFn) {
	log.Debug("Installing statusz checkers")
	mux.Handle(path, registerStatuszCheck(statusChecker))
}

func registerStatuszCheck(statusChecker statusCheckFn) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		status := statusChecker()
		if status == "" {
			w.WriteHeader(http.StatusOK)
		}
		http.Error(w, status, http.StatusServiceUnavailable)
	})
}
