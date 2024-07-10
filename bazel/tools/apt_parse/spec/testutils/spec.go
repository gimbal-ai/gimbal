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

package testutils

import (
	"fmt"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
)

// PinnedPackage is a testing version of `spec.PinnedPackage`.
// It allows specifying DirectDependencies by name instead of as a pointer to another `PinnedPackage`.
// This makes declaring test cases involving `spec.PinnedPackage` much easier.
type PinnedPackage struct {
	Name               string
	Version            string
	Arch               string
	URLs               []string
	SHA256             string
	RepoName           string
	DirectDependencies []string
	ExcludePaths       []string
	ExtraSymlinks      map[string]string
}

// ConvertToSpecPackages converts from a list of testutils.PinnedPackage to a list of spec.PinnedPackage.
// DirectDependencies listed by name in each testuils.PinnedPackage are resolved from the other packages in testPkgs.
func ConvertToSpecPackages(testPkgs []*PinnedPackage) ([]*spec.PinnedPackage, error) {
	nameToIndex := make(map[string]int)
	for i, p := range testPkgs {
		nameToIndex[p.Name] = i
	}

	pkgs := make([]*spec.PinnedPackage, len(testPkgs))
	// First create all the pinned packages.
	for i, p := range testPkgs {
		pkgs[i] = &spec.PinnedPackage{
			Name:               p.Name,
			Version:            p.Version,
			Arch:               p.Arch,
			URLs:               p.URLs,
			SHA256:             p.SHA256,
			RepoName:           p.RepoName,
			ExcludePaths:       p.ExcludePaths,
			DirectDependencies: make([]*spec.PinnedPackage, 0, len(p.DirectDependencies)),
		}
	}
	// Then set the direct dependencies and symlinks for each one.
	for i, p := range testPkgs {
		for _, dep := range p.DirectDependencies {
			ind, ok := nameToIndex[dep]
			if !ok {
				return nil, fmt.Errorf("dependency %s not in list of expected dependencies", dep)
			}
			pkgs[i].DirectDependencies = append(pkgs[i].DirectDependencies, pkgs[ind])
		}
		for s, t := range p.ExtraSymlinks {
			pkgs[i].ExtraSymlinks = append(pkgs[i].ExtraSymlinks, &spec.Symlink{Source: s, Target: t})
		}
	}
	return pkgs, nil
}
