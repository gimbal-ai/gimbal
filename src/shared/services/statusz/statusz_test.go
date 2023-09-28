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

package statusz_test

import (
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/shared/services/statusz"
)

func TestInstallPathHandler_NotReady(t *testing.T) {
	mux := http.NewServeMux()
	statusz.InstallPathHandler(mux, "/statusz/test", func() string {
		return "thisIsATest"
	})
	req, err := http.NewRequest("GET", "http://abc.com/statusz/test", nil)
	require.NoError(t, err)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	assert.Equal(t, http.StatusServiceUnavailable, w.Code)
	assert.Equal(t, w.Body.String(), "thisIsATest\n")
}

func TestInstallPathHandler_OK(t *testing.T) {
	mux := http.NewServeMux()
	statusz.InstallPathHandler(mux, "/statusz/test", func() string {
		return ""
	})
	req, err := http.NewRequest("GET", "http://abc.com/statusz/test", nil)
	require.NoError(t, err)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, w.Body.String(), "\n")
}
