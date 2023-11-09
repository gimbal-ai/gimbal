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

package main

import (
	"compress/gzip"
	"fmt"
	"io"
	"os"
	"strings"

	log "github.com/sirupsen/logrus"
	"github.com/spf13/cobra"
	"github.com/spf13/viper"
	"github.com/ulikunitz/xz"
)

var rootCmd = &cobra.Command{
	Use:   "root",
	Short: "root command used by other commands",
}

var satisfyCmd = &cobra.Command{
	Use:   "satisfy",
	Short: "satisfy dependencies for the given specs",
	Run: func(cmd *cobra.Command, args []string) {
		_ = viper.BindPFlags(cmd.Flags())
		satisfyDeps()
	},
}

func init() {
	satisfyCmd.Flags().String("download_cache_dir", "", "Directory containing cached downloads")
	satisfyCmd.Flags().StringSlice("specs", []string{}, "List of yaml files specifying packages to include/exclude from the sysroot and where to get them from")
	satisfyCmd.Flags().String("arch", "", "Architecture to build sysroot for")

	rootCmd.AddCommand(satisfyCmd)
}

func packageDBReader(d *downloader, url string) (io.Reader, func(), error) {
	contents, err := d.Download(url)
	if err != nil {
		return nil, func() {}, err
	}
	cleanup := func() { contents.Close() }
	var r io.Reader
	r = contents
	if strings.HasSuffix(url, "xz") {
		r, err = xz.NewReader(r)
		if err != nil {
			return nil, cleanup, err
		}
	} else if strings.HasSuffix(url, "gz") {
		r, err = gzip.NewReader(r)
		if err != nil {
			return nil, cleanup, err
		}
	}
	return r, cleanup, nil
}

func satisfyDeps() {
	log.SetOutput(os.Stderr)

	downloadCacheDir := viper.GetString("download_cache_dir")
	if downloadCacheDir == "" {
		log.Fatal("must specify download_cache_dir")
	}
	specs := viper.GetStringSlice("specs")
	if len(specs) == 0 {
		log.Fatal("must specify at least one spec")
	}
	arch := viper.GetString("arch")
	if arch == "" {
		log.Fatal("must specify arch")
	}

	d := &downloader{
		dir: downloadCacheDir,
	}

	combinedSpec, err := parseAndCombineSpecs(specs)
	if err != nil {
		log.WithError(err).Fatal("failed to parse specs")
	}
	filteredSpec := combinedSpec.removeIncludesFromExclude()

	db := newPackageDatabase()

	for _, dbSpec := range combinedSpec.PackageDatabases {
		archSpec, ok := dbSpec.Architectures[arch]
		if !ok {
			continue
		}
		r, cleanup, err := packageDBReader(d, archSpec.IndexURL)
		defer cleanup()
		if err != nil {
			log.WithError(err).Fatal("failed to download package database index")
		}
		if err := db.parsePackageDatabase(r, archSpec.DownloadPrefix); err != nil {
			log.WithError(err).Fatal("failed to parse package database")
		}
	}
	dependencySatisfier := newDepSatisfier(db, filteredSpec)
	debs, err := dependencySatisfier.listRequiredDebs()
	if err != nil {
		log.WithError(err).Fatal("failed to find all required packages")
	}
	for _, url := range debs {
		fmt.Println(url)
	}
}

func main() {
	if err := rootCmd.Execute(); err != nil {
		log.WithError(err).Fatal("failed to execute package_satisfier")
	}
}
