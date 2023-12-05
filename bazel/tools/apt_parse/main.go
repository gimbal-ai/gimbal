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
	"errors"
	"fmt"
	"os"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/pflag"
	"github.com/spf13/viper"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/bazelgen"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/index"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/resolve"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
)

func init() {
	pflag.StringSlice("spec", []string{}, "Apt specs to parse into bzl repo definitions")
	pflag.StringSlice("arch", []string{}, "List of architectures to find package dependencies for")
	pflag.String("out_bzl", "", "Path to output .bzl file")
	pflag.String("macro_name", "deb_repos", "Name for bazel macro which defines the deb repos")
	pflag.Parse()
	_ = viper.BindPFlags(pflag.CommandLine)
}

func generate(d index.Downloader, specs []string, archs []string, macroName string, outPath string) error {
	gen, err := bazelgen.NewBazelGenerator(macroName)
	if err != nil {
		return err
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
			gen.AddPinnedSet(pps)
		}
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
