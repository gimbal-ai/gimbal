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
	"postgres:15-alpine":                       "gml/src/testutils/postgres_15_alpine/tarball.tar",
	"victoriametrics/victoria-metrics:v1.93.6": "gml/src/testutils/victoria_metrics_v1_93_6/tarball.tar",
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
	runfiles := os.Getenv("BAZEL_RUNFILES")
	if runfiles != "" {
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
