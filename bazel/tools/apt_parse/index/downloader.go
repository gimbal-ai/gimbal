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

package index

import (
	"compress/gzip"
	"io"
	"net/http"
	"strings"

	"github.com/ulikunitz/xz"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
)

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
