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

package spec

import (
	"io"

	"gopkg.in/yaml.v3"
)

// PackageSet represents a set of apt package dependencies to resolve.
// This can be thought of as the equivalent of python's requirements.txt but for apt packages.
// The apt_parse tool takes PackageSets and converts them into a listing of all
// transitive dependencies needed, in the form of a list of PinnedPackages.
// All such dependencies are resolved from the apt repositories listed in the PackageSet.
type PackageSet struct {
	// Repositories to pull package information from.
	Repositories []*Repository `yaml:"repositories"`
	Packages     []*Package    `yaml:"packages"`
	// ManualPackages are added unmodified to the output. This can be used to pull in `.deb` files
	// that don't have a corresponding apt repository, or to pull in a package without any of its transitive deps.
	ManualPackages []*PinnedPackage `yaml:"manual_packages"`
	// ExcludePackages excludes any packages listed from the transitive deps.
	// Dependency search stops when an excluded package is reached,
	// so it will also exclude the transitive dependencies of the excluded packages.
	ExcludePackages []string `yaml:"exclude_packages"`
	// ExcludePaths are paths to exclude from the generated bazel repos for the packages in this set.
	// This is useful when, for example, there is a file that has a non-UTF8 path that bazel can't interpret.
	ExcludePaths []string `yaml:"exclude_paths"`
}

// Package represents a single direct dependency on a package.
type Package struct {
	Name    string `yaml:"name"`
	Version string `yaml:"version"`
}

// Repository represents a debian-style package repository with package information contained in a "Package" index file.
type Repository struct {
	Name string `yaml:"name"`
	// Arch is the architecture for the packages in this repository.
	Arch string `yaml:"arch"`
	// IndexURL is the url to download the debian-style package index.
	IndexURL string `yaml:"index_url"`
	// DownloadPrefix to prepend to `filename` from debian control information in the index.
	DownloadPrefix string `yaml:"download_prefix"`
}

// PinnedPackage represents a fully-resolved package with all its dependencies also resolved.
type PinnedPackage struct {
	Name               string
	Version            string
	Arch               string
	URL                string
	SHA256             string
	RepoName           string
	DirectDependencies []*PinnedPackage
	ExcludePaths       []string
}

// ParsePackageSet parses a yaml file into a PackageSet.
func ParsePackageSet(r io.Reader) (*PackageSet, error) {
	dec := yaml.NewDecoder(r)
	dec.KnownFields(true)
	ps := &PackageSet{}
	if err := dec.Decode(ps); err != nil {
		return nil, err
	}
	return ps, nil
}
