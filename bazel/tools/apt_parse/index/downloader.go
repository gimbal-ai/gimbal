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

package index

import (
	"compress/gzip"
	"errors"
	"io"
	"net/http"
	"strings"

	"github.com/ulikunitz/xz"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
)

// ErrDownloadFailed is returned when a download returns a non 200 code.
var ErrDownloadFailed = errors.New("download status not OK")

// Downloader is the interface for downloading repository specs.
type Downloader interface {
	// Download downloads a repository index from the given spec.
	Download(*spec.Repository) (*Index, error)
}

// NewHTTPDownloader returns a downloader that uses http.Get to get the repository index file.
func NewHTTPDownloader() Downloader {
	return &httpDownloaderImpl{}
}

type httpDownloaderImpl struct{}

func (*httpDownloaderImpl) Download(repo *spec.Repository) (*Index, error) {
	resp, err := http.Get(repo.IndexURL)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, ErrDownloadFailed
	}
	defer resp.Body.Close()
	var r io.Reader
	r = resp.Body
	if strings.HasSuffix(repo.IndexURL, "xz") {
		r, err = xz.NewReader(r)
		if err != nil {
			return nil, err
		}
	} else if strings.HasSuffix(repo.IndexURL, "gz") {
		r, err = gzip.NewReader(r)
		if err != nil {
			return nil, err
		}
	}
	index, err := NewIndex(r, repo)
	if err != nil {
		return nil, err
	}
	return index, nil
}
