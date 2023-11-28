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

package handler

import (
	"errors"
	"net/http"

	"gimletlabs.ai/gimlet/src/shared/services/env"
)

// Error represents a handler error. It provides methods for a HTTP status
// code and embeds the built-in error interface.
type Error interface {
	error
	Status() int
}

// StatusError represents an error with an associated HTTP status code.
type StatusError struct {
	Code int
	Err  error
}

// Error returns the status errors underlying error.
func (se StatusError) Error() string {
	return se.Err.Error()
}

// Status returns the HTTP status code.
func (se StatusError) Status() int {
	return se.Code
}

// NewStatusError creates a new status error with a code and msg.
func NewStatusError(code int, msg string) *StatusError {
	return &StatusError{
		Code: code,
		Err:  errors.New(msg),
	}
}

// Handler struct that takes a configured BaseEnv and a function matching
// our signature.
type Handler struct {
	env env.Env
	H   func(w http.ResponseWriter, r *http.Request) error
}

// New returns a new App handler.
func New(e env.Env, f func(w http.ResponseWriter, r *http.Request) error) *Handler {
	return &Handler{e, f}
}

// ServeHTTP allows our Handler type to satisfy http.Handler.
func (h Handler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	rc := r.WithContext(env.NewContext(r.Context(), h.env))
	err := h.H(w, rc)
	if err != nil {
		var e Error
		switch {
		case errors.As(err, &e):
			http.Error(w, e.Error(), e.Status())
		default:
			// Any error types we don't specifically look out for default
			// to serving a HTTP 500
			http.Error(w, http.StatusText(http.StatusInternalServerError),
				http.StatusInternalServerError)
		}
	}
}
