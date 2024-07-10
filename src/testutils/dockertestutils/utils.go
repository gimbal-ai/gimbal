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

package dockertestutils

import (
	"errors"
	"fmt"
	"os"

	"github.com/bazelbuild/rules_go/go/runfiles"
	"github.com/ory/dockertest/v3"
	"github.com/ory/dockertest/v3/docker"
	log "github.com/sirupsen/logrus"
)

var imageToTarget = map[string]string{
	"victoriametrics/victoria-metrics:v1.93.6": "gml/src/testutils/victoria_metrics_v1_93_6/tarball.tar",
	"pgvector/pgvector:pg15":                   "gml/src/testutils/pgvector_pg15/tarball.tar",
}

var (
	ErrUnknownImage    = errors.New("must pull image with oci.pull and add to the map in the dockertestutils package")
	ErrBadImage        = errors.New("expected an image of the form repository:tag")
	ErrMissingRunfile  = errors.New("failed to find image runfile")
	ErrCannotReadImage = errors.New("failed to read image tarball")
)

func LoadOrFetchImage(pool *dockertest.Pool, imageRepo string, imageTag string) error {
	imageTarget := imageToTarget[fmt.Sprintf("%s:%s", imageRepo, imageTag)]
	if imageTarget == "" {
		return ErrUnknownImage
	}
	runfilesEnv := os.Getenv("BAZEL_RUNFILES")
	if runfilesEnv != "" {
		return loadImageFromRunfiles(pool, imageTarget)
	}
	return fetchImage(pool, imageRepo, imageTag)
}

func loadImageFromRunfiles(pool *dockertest.Pool, imageTarget string) error {
	imgPath, err := runfiles.Rlocation(imageTarget)
	if err != nil {
		return ErrMissingRunfile
	}
	f, err := os.Open(imgPath)
	if err != nil {
		return ErrCannotReadImage
	}
	return pool.Client.LoadImage(docker.LoadImageOptions{
		InputStream: f,
	})
}

func fetchImage(pool *dockertest.Pool, imageRepo string, imageTag string) error {
	// Check if image already exists before pulling.
	_, err := pool.Client.InspectImage(fmt.Sprintf("%s:%s", imageRepo, imageTag))
	if err == nil {
		return nil
	}

	log.Warn("fetching docker image from the network. this might cause test slowness/flakes")
	return pool.Client.PullImage(docker.PullImageOptions{
		Repository: imageRepo,
		Tag:        imageTag,
	}, docker.AuthConfiguration{})
}
