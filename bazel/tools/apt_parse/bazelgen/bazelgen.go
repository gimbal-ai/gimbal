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

package bazelgen

import (
	"fmt"
	"sort"
	"strings"

	"github.com/bazelbuild/bazel-gazelle/rule"
	"github.com/bazelbuild/buildtools/build"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
)

// BazelGenerator generates a bzl file with a single bazel macro with a list of repository rules, one for each `PinnedPackage` given to the generator.
type BazelGenerator struct {
	file  *rule.File
	names map[string]bool
}

const (
	debLoad        = "//bazel/rules_pkg:pkg_provider_archives.bzl"
	debRule        = "deb_archive_w_pkg_providers"
	allFilesTarget = ":all_files"
)

// NewBazelGenerator returns a new BazelGenerator that generates a bzl file with a macro with the given name.
func NewBazelGenerator(macroName string) (*BazelGenerator, error) {
	file, err := rule.LoadMacroData("", "", macroName, []byte{})
	if err != nil {
		return nil, err
	}
	l := rule.NewLoad(debLoad)
	l.Add(debRule)
	l.Insert(file, 0)

	return &BazelGenerator{
		file:  file,
		names: make(map[string]bool),
	}, nil
}

// AddPinnedSet adds the given PinnedPackages to the bazel macro.
func (g *BazelGenerator) AddPinnedSet(pkgs []*spec.PinnedPackage) {
	for _, p := range pkgs {
		name := bazelRepoName(p)
		if g.names[name] {
			continue
		}
		g.names[name] = true
		g.addRule(name, p)
	}
}

// Content returns the bzl file content as a string.
func (g *BazelGenerator) Content() string {
	g.file.SortMacro()
	g.file.Sync()
	return string(build.Format(g.file.File))
}

// Save saves the bzl file to the given path.
func (g *BazelGenerator) Save(path string) error {
	g.file.SortMacro()
	return g.file.Save(path)
}

func (g *BazelGenerator) addRule(name string, p *spec.PinnedPackage) {
	r := rule.NewRule(debRule, name)
	r.SetAttr("name", name)
	r.SetAttr("urls", p.URLs)
	r.SetAttr("sha256", p.SHA256)
	depMap := make(map[string]string)
	for _, dep := range p.DirectDependencies {
		depName := bazelRepoName(dep)
		depMap[depName] = fmt.Sprintf("@%s//%s", depName, allFilesTarget)
	}
	deps := make([]string, 0, len(depMap))
	for _, dep := range depMap {
		deps = append(deps, dep)
	}
	sort.Strings(deps)
	r.SetAttr("deps", deps)
	if p.ExcludePaths != nil && len(p.ExcludePaths) > 0 {
		r.SetAttr("exclude_paths", p.ExcludePaths)
	}
	if p.ExtraSymlinks != nil && len(p.ExtraSymlinks) > 0 {
		extraSymlinks := make(map[string]string)
		for _, sl := range p.ExtraSymlinks {
			extraSymlinks[sl.Source] = sl.Target
		}
		r.SetAttr("extra_symlinks", extraSymlinks)
	}
	r.Insert(g.file)
}

func bazelRepoName(p *spec.PinnedPackage) string {
	repoName := strings.Join([]string{
		p.RepoName,
		p.Name,
		p.Arch,
	}, "_")
	return sanitize(repoName)
}

func sanitize(repoName string) string {
	// From bazel: repo names may contain only A-Z, a-z, 0-9, '-', '_', '.' and '~' and must not start with '~'
	return strings.Map(func(r rune) rune {
		if (r >= 'A' && r <= 'Z') ||
			(r >= 'a' && r <= 'z') ||
			(r >= '0' && r <= '9') ||
			r == '-' ||
			r == '_' ||
			r == '.' ||
			r == '~' {
			return r
		}
		return '_'
	}, repoName)
}
