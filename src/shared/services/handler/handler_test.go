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

package handler_test

import (
	"errors"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/src/shared/services/env"
	"gimletlabs.ai/gimlet/src/shared/services/handler"
)

func TestHandler_ServeHTTP(t *testing.T) {
	testHandler := func(w http.ResponseWriter, r *http.Request) error {
		return nil
	}
	h := handler.New(env.New("test", ""), testHandler)

	req, err := http.NewRequest("GET", "http://test.com/", nil)
	require.NoError(t, err)

	rw := httptest.NewRecorder()
	h.ServeHTTP(rw, req)
	assert.Equal(t, http.StatusOK, rw.Code)
}

func TestHandler_ServeHTTP_StatusError(t *testing.T) {
	testHandler := func(w http.ResponseWriter, r *http.Request) error {
		return &handler.StatusError{http.StatusUnauthorized, errors.New("badness")}
	}
	h := handler.New(env.New("test", ""), testHandler)

	req, err := http.NewRequest("GET", "http://test.com/", nil)
	require.NoError(t, err)

	rw := httptest.NewRecorder()
	h.ServeHTTP(rw, req)
	assert.Equal(t, http.StatusUnauthorized, rw.Code)
	assert.Equal(t, "badness\n", rw.Body.String())
}

func TestHandler_ServeHTTP_RegularError(t *testing.T) {
	testHandler := func(w http.ResponseWriter, r *http.Request) error {
		return errors.New("badness")
	}
	h := handler.New(env.New("test", ""), testHandler)

	req, err := http.NewRequest("GET", "http://test.com/", nil)
	require.NoError(t, err)

	rw := httptest.NewRecorder()
	h.ServeHTTP(rw, req)
	assert.Equal(t, http.StatusInternalServerError, rw.Code)
	assert.Equal(t, "Internal Server Error\n", rw.Body.String())
}
