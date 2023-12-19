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

package mirror

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"cloud.google.com/go/storage"
	"github.com/cenkalti/backoff/v4"
	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"
	"golang.org/x/sync/errgroup"
	"google.golang.org/api/iterator"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
)

func init() {
	pflag.Bool("check_shas", false, "Whether to check that the shas are correct on the mirrors")
}

// Mirrorer is the interface for a package mirrorer. A package mirrorer accepts lists of pinned packages,
// mirrors the package, and then updates the pinned package definition to include the mirror.
type Mirrorer interface {
	Mirror([]*spec.PinnedPackage) error
	Wait() error
}

type noopMirrorer struct{}

// NewNoopMirrorer creates a mirrorer that does nothing.
func NewNoopMirrorer() Mirrorer {
	return &noopMirrorer{}
}

func (*noopMirrorer) Mirror([]*spec.PinnedPackage) error {
	return nil
}

func (*noopMirrorer) Wait() error {
	return nil
}

type gcsMirrorer struct {
	bucket string
	c      *storage.Client
	path   string

	eg errgroup.Group

	mirroredSHAs map[string]bool
}

// NewGCSMirrorer creates a mirroer that mirrors packages to a GCS bucket.
func NewGCSMirrorer(bucket string, path string) (Mirrorer, error) {
	ctx := context.Background()
	client, err := storage.NewClient(ctx)
	if err != nil {
		return nil, err
	}
	m := &gcsMirrorer{
		c:            client,
		bucket:       bucket,
		path:         path,
		mirroredSHAs: make(map[string]bool),
	}
	m.eg.SetLimit(32)
	if err := m.start(); err != nil {
		return nil, err
	}
	return m, nil
}

func (m *gcsMirrorer) Mirror(pps []*spec.PinnedPackage) error {
	for _, p := range pps {
		pkg := p
		m.eg.Go(func() error {
			return m.mirrorIfNeeded(pkg)
		})
		p.URLs = append([]string{mirrorURL(m.bucket, m.path, p.SHA256)}, p.URLs...)
	}
	return nil
}

func (m *gcsMirrorer) shaFromName(name string) string {
	shaAndSuffix := strings.TrimPrefix(name, m.path+"/")
	return strings.TrimSuffix(shaAndSuffix, ".deb")
}

func (m *gcsMirrorer) start() error {
	iter := m.c.Bucket(m.bucket).Objects(context.Background(), &storage.Query{
		Prefix: m.path,
	})
	attrs, err := iter.Next()
	for err == nil {
		m.mirroredSHAs[m.shaFromName(attrs.Name)] = true
		attrs, err = iter.Next()
	}
	if err != nil && !errors.Is(err, iterator.Done) {
		return err
	}
	if viper.GetBool("check_shas") {
		if err := m.checkSHAs(); err != nil {
			return err
		}
	}
	return nil
}

func (m *gcsMirrorer) Wait() error {
	return m.eg.Wait()
}

func (m *gcsMirrorer) mirrorIfNeeded(p *spec.PinnedPackage) error {
	if m.mirroredSHAs[p.SHA256] {
		return nil
	}
	ctx := context.Background()
	bo := backoff.NewExponentialBackOff()
	bo.MaxElapsedTime = 5 * time.Minute
	err := backoff.Retry(func() error {
		return m.mirror(ctx, p)
	}, bo)
	if err != nil {
		return err
	}
	return nil
}

const gcsPrefix string = "https://storage.googleapis.com"

func mirrorURL(bucket, path, sha string) string {
	return fmt.Sprintf("%s/%s/%s/%s.deb", gcsPrefix, bucket, path, sha)
}

func (m *gcsMirrorer) mirror(ctx context.Context, p *spec.PinnedPackage) error {
	log.Infof("Mirroring %s", p.Name)
	origURL := ""
	// TODO(james): support multiple URLs.
	if len(p.URLs) > 2 {
		log.Warn("GCSMirrorer doesn't currently support packages with more than 1 non-mirror URL")
	}
	for _, url := range p.URLs {
		if strings.HasPrefix(url, gcsPrefix) {
			continue
		}
		origURL = url
		break
	}
	if origURL == "" {
		return backoff.Permanent(fmt.Errorf("no valid URLs to mirror in package: %v", p))
	}
	resp, err := http.Get(origURL)
	if err != nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download of %s failed", origURL)
	}
	defer resp.Body.Close()
	objPath := fmt.Sprintf("%s/%s.deb", m.path, p.SHA256)
	w := m.c.Bucket(m.bucket).Object(objPath).NewWriter(ctx)
	defer w.Close()
	if _, err := io.Copy(w, resp.Body); err != nil {
		return err
	}
	if err := w.Close(); err != nil {
		return err
	}
	m.mirroredSHAs[p.SHA256] = true
	return nil
}

func (m *gcsMirrorer) checkSHAs() error {
	eg := errgroup.Group{}
	for sha := range m.mirroredSHAs {
		s := sha
		eg.Go(func() error {
			objPath := fmt.Sprintf("%s/%s.deb", m.path, s)
			obj := m.c.Bucket(m.bucket).Object(objPath)
			r, err := obj.NewReader(context.Background())
			if err != nil {
				return err
			}
			h := sha256.New()
			if _, err := io.Copy(h, r); err != nil {
				return err
			}
			if fmt.Sprintf("%x", h.Sum(nil)) != s {
				log.Infof("SHA mismatch for %s, deleting", s)
				m.mirroredSHAs[s] = false
				if err := obj.Delete(context.Background()); err != nil {
					return err
				}
			}
			return nil
		})
	}
	return eg.Wait()
}

type fakeGCSMirrorer struct {
	bucket string
	path   string
}

// NewFakeGCSMirrorer returns a Mirrorer that updates the pinned packages in the same way as the GCSMirrorer,
// but doesn't actually mirror to GCS.
func NewFakeGCSMirrorer(bucket string, path string) Mirrorer {
	return &fakeGCSMirrorer{
		bucket: bucket,
		path:   path,
	}
}

func (m *fakeGCSMirrorer) Mirror(pps []*spec.PinnedPackage) error {
	for _, p := range pps {
		p.URLs = append([]string{mirrorURL(m.bucket, m.path, p.SHA256)}, p.URLs...)
	}
	return nil
}

func (*fakeGCSMirrorer) Wait() error {
	return nil
}
