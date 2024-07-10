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

package main

import (
	"errors"
	"fmt"
	"os"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/bazelgen"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/index"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/mirror"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/resolve"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
)

func init() {
	pflag.StringSlice("spec", []string{}, "Apt specs to parse into bzl repo definitions")
	pflag.StringSlice("arch", []string{}, "List of architectures to find package dependencies for")
	pflag.String("out_bzl", "", "Path to output .bzl file")
	pflag.String("macro_name", "deb_repos", "Name for bazel macro which defines the deb repos")
	pflag.String("mirror_bucket", "gimlet-dev-infra-public", "Bucket to mirror debs to")
	pflag.String("mirror_path", "deb-mirrors", "Bucket to mirror debs to")
	pflag.Bool("fake_mirroring", false, "If true, skips mirroring but outputs a file as if mirrors were created. Useful for checking if the file is up to date without running any of the mirroring.")
	pflag.Parse()
	_ = viper.BindPFlags(pflag.CommandLine)
}

func generate(d index.Downloader, specs []string, archs []string, macroName string, outPath string) error {
	gen, err := bazelgen.NewBazelGenerator(macroName)
	if err != nil {
		return err
	}
	m := mirror.NewNoopMirrorer()
	mirrorBucket := viper.GetString("mirror_bucket")
	mirrorPath := viper.GetString("mirror_path")
	if mirrorBucket != "" && mirrorPath != "" {
		if viper.GetBool("fake_mirroring") {
			m = mirror.NewFakeGCSMirrorer(mirrorBucket, mirrorPath)
		} else {
			var err error
			m, err = mirror.NewGCSMirrorer(mirrorBucket, mirrorPath)
			if err != nil {
				return err
			}
		}
	}
	for _, specPath := range specs {
		f, err := os.Open(specPath)
		if err != nil {
			return err
		}
		defer f.Close()
		ps, err := spec.ParsePackageSet(f)
		if err != nil {
			return fmt.Errorf("failed to parse spec %s: %w", specPath, err)
		}
		for _, arch := range archs {
			r, err := resolve.NewResolver(d, ps, arch)
			if errors.Is(err, resolve.ErrNoPackagesForArch) {
				continue
			}
			if err != nil {
				return err
			}
			pps, err := r.Resolve()
			if err != nil {
				return fmt.Errorf("failed to resolve spec %s: %w", specPath, err)
			}
			if err := m.Mirror(pps); err != nil {
				return err
			}
			gen.AddPinnedSet(pps)
		}
	}
	if err := m.Wait(); err != nil {
		return err
	}
	return gen.Save(outPath)
}

func main() {
	specs := viper.GetStringSlice("spec")
	if len(specs) == 0 {
		log.Fatal("Must specify at least one spec")
	}
	archs := viper.GetStringSlice("arch")
	if len(archs) == 0 {
		log.Fatal("Must specify at least one architecture")
	}
	outPath := viper.GetString("out_bzl")
	if outPath == "" {
		log.Fatal("Must specify --out_bzl")
	}
	macroName := viper.GetString("macro_name")

	d := index.NewHTTPDownloader()
	if err := generate(d, specs, archs, macroName, outPath); err != nil {
		log.WithError(err).Fatal("failed to create bazel repo definitions from apt specs")
	}
}
