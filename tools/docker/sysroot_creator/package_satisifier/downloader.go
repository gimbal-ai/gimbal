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
	"crypto/sha256"
	"fmt"
	"io"
	"net/http"
	"os"
	"path"
)

type downloader struct {
	dir string
}

func (d *downloader) Download(url string) (io.ReadCloser, error) {
	cachePath := d.cachePath(url)
	if _, err := os.Stat(cachePath); err != nil && os.IsNotExist(err) {
		if err := d.downloadTo(url, cachePath); err != nil {
			return nil, err
		}
	} else if err != nil {
		return nil, err
	}

	f, err := os.Open(cachePath)
	if err != nil {
		return nil, err
	}
	return f, nil
}

func (d *downloader) cachePath(url string) string {
	sha := sha256.Sum256([]byte(url))
	return path.Join(d.dir, fmt.Sprintf("%x", sha))
}

func (d *downloader) downloadTo(url string, outPath string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("received non-OK HTTP status: %s", resp.Status)
	}
	contents, err := io.ReadAll(resp.Body)
	if err != nil {
		return err
	}
	if err := os.WriteFile(outPath, contents, 0o655); err != nil {
		return err
	}
	return nil
}
