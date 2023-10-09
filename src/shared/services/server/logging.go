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
	"bufio"
	"fmt"
	"net"
	"net/http"
	"os"
	"time"

	log "github.com/sirupsen/logrus"
)

// SetupServiceLogging sets up a consistent logging env for all services.
func SetupServiceLogging() {
	// Setup logging.
	log.SetOutput(os.Stdout)
	log.SetLevel(log.InfoLevel)
}

// HTTPLoggingMiddleware is a middleware function used for logging HTTP requests.
func HTTPLoggingMiddleware(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t := time.Now()

		wr := newLogInterceptWriter(w, r, t)

		h.ServeHTTP(wr, r)

		log.WithFields(log.Fields{
			"http_req_client_ip":  wr.reqClientIP,
			"http_req_duration":   time.Since(t).Milliseconds(),
			"http_req_host":       wr.reqHost,
			"http_req_method":     wr.reqMethod,
			"http_req_path":       wr.reqPath,
			"http_req_protocol":   wr.reqProto,
			"http_req_size":       wr.reqSize,
			"http_req_time":       wr.reqTime,
			"http_req_user_agent": wr.reqUserAgent,
			"http_res_size":       wr.resSize,
			"http_res_status":     wr.resStatus,
		}).Info()
	})
}

// Acts as an adapter for http.ResponseWriter type to store request and
// response data.
type logInterceptWriter struct {
	http.ResponseWriter

	reqClientIP  string
	reqHost      string
	reqMethod    string
	reqPath      string
	reqProto     string
	reqSize      int64 // bytes
	reqTime      string
	reqUserAgent string

	resStatus int
	resSize   int // bytes
}

func newLogInterceptWriter(w http.ResponseWriter, r *http.Request, t time.Time) *logInterceptWriter {
	return &logInterceptWriter{
		ResponseWriter: w,

		reqClientIP:  r.Header.Get("X-Forwarded-For"),
		reqMethod:    r.Method,
		reqHost:      r.Host,
		reqPath:      r.RequestURI,
		reqProto:     r.Proto,
		reqSize:      r.ContentLength,
		reqTime:      t.Format(time.RFC3339),
		reqUserAgent: r.UserAgent(),
	}
}

// WriteHeader overrides http.ResponseWriter type.
func (w *logInterceptWriter) WriteHeader(status int) {
	if w.resStatus == 0 {
		w.resStatus = status
		w.ResponseWriter.WriteHeader(status)
	}
}

// Write overrides http.ResponseWriter type.
func (w *logInterceptWriter) Write(body []byte) (int, error) {
	if w.resStatus == 0 {
		w.WriteHeader(http.StatusOK)
	}

	var err error
	w.resSize, err = w.ResponseWriter.Write(body)

	return w.resSize, err
}

// Flush overrides http.Flusher type.
func (w *logInterceptWriter) Flush() {
	if fl, ok := w.ResponseWriter.(http.Flusher); ok {
		if w.resStatus == 0 {
			w.WriteHeader(http.StatusOK)
		}

		fl.Flush()
	}
}

// Hijack overrides http.Hijacker type.
func (w *logInterceptWriter) Hijack() (net.Conn, *bufio.ReadWriter, error) {
	hj, ok := w.ResponseWriter.(http.Hijacker)
	if !ok {
		return nil, nil, fmt.Errorf("the hijacker interface is not supported")
	}

	return hj.Hijack()
}
