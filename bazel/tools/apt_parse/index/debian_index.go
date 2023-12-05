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
	"bufio"
	"fmt"
	"io"

	"pault.ag/go/debian/control"
	"pault.ag/go/debian/dependency"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
)

// Package represents a debian-style package with debian's control information.
type Package struct {
	BinaryIndex control.BinaryIndex
	Provides    []string
	Repo        *spec.Repository
}

// PackageOption represents a possibility for fulfilling a package dependency.
type PackageOption struct {
	Pkg *Package
	// Virtual returns whether this is a provider of the named virtual package.
	Virtual bool
}

// Index represents an debian-style index of packages. It holds the dependencies between packages.
type Index struct {
	packages map[string][]*PackageOption
}

// NewIndex returns a new debian index from the given index file.
// See here for the spec of the index file: https://wiki.debian.org/DebianRepository/Format#A.22Packages.22_Indices.
func NewIndex(r io.Reader, repo *spec.Repository) (*Index, error) {
	br := bufio.NewReader(r)
	indices, err := control.ParseBinaryIndex(br)
	if err != nil {
		return nil, err
	}
	ind := &Index{
		packages: make(map[string][]*PackageOption),
	}

	for _, bi := range indices {
		provides, err := parseProvidesFromIndex(bi)
		if err != nil {
			return nil, err
		}
		p := &Package{
			BinaryIndex: bi,
			Provides:    provides,
			Repo:        repo,
		}
		ind.packages[bi.Package] = append(ind.packages[bi.Package], &PackageOption{
			Pkg:     p,
			Virtual: false,
		})
		for _, virtual := range p.Provides {
			ind.packages[virtual] = append(ind.packages[virtual], &PackageOption{
				Pkg:     p,
				Virtual: true,
			})
		}
	}

	return ind, nil
}

// HasPackage returns whether the index has the given package.
func (i *Index) HasPackage(name string) bool {
	_, ok := i.packages[name]
	return ok
}

// PackageOptions returns all available options for a package by the given name.
// For example, different versions of the same package, or different providers of a virtual package.
func (i *Index) PackageOptions(name string) []*PackageOption {
	return i.packages[name]
}

func parseProvidesFromIndex(index control.BinaryIndex) ([]string, error) {
	provideVal, ok := index.Values["Provides"]
	if !ok {
		return []string{}, nil
	}
	providesDep, err := dependency.Parse(provideVal)
	if err != nil {
		return nil, err
	}
	provides := []string{}
	for _, rel := range providesDep.Relations {
		if len(rel.Possibilities) != 1 {
			return nil, fmt.Errorf("unexpected OR relation in Provides")
		}
		provides = append(provides, rel.Possibilities[0].Name)
	}
	return provides, nil
}
