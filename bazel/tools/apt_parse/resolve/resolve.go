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

package resolve

import (
	"errors"
	"fmt"
	"sort"
	"strings"

	"pault.ag/go/debian/dependency"
	"pault.ag/go/debian/version"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/index"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
)

// Resolver resolves PackageSets into a list of `spec.PinnedPackages` for all transitive dependencies listed in the `PackageSet`.
type Resolver struct {
	ps         *spec.PackageSet
	manualPkgs []*spec.PinnedPackage

	indices  []*index.Index
	resolved map[string]*spec.PinnedPackage
	excludes map[string]bool
}

// ErrNoPackagesForArch is returned when a resolver's package set doesn't have any packages for the requested architecture.
var ErrNoPackagesForArch = errors.New("package set has no packages for given architecture")

// NewResolver returns a new resolver for the given architecture and package set.
// If there are no packages in the package set for the given architecture, it returns ErrNoPackagesForArch.
func NewResolver(downloader index.Downloader, ps *spec.PackageSet, arch string) (*Resolver, error) {
	r := &Resolver{
		ps:         ps,
		manualPkgs: make([]*spec.PinnedPackage, 0),
		indices:    make([]*index.Index, 0),
		resolved:   make(map[string]*spec.PinnedPackage),
		excludes:   make(map[string]bool),
	}
	for _, repo := range ps.Repositories {
		if repo.Arch != arch {
			continue
		}
		ind, err := downloader.Download(repo)
		if err != nil {
			return nil, err
		}
		r.indices = append(r.indices, ind)
	}
	for _, p := range r.ps.ManualPackages {
		if p.Arch == arch {
			r.manualPkgs = append(r.manualPkgs, p)
		}
	}
	if len(r.indices) == 0 && len(r.manualPkgs) == 0 {
		return nil, ErrNoPackagesForArch
	}
	for _, exc := range ps.ExcludePackages {
		r.excludes[exc] = true
	}
	return r, nil
}

// Resolve runs the resolver returning the list of transitive package dependencies.
func (r *Resolver) Resolve() ([]*spec.PinnedPackage, error) {
	dfs := NewDFS[*graphNode]()
	for _, p := range r.ps.Packages {
		pkg := r.packageFromPossibility(dependency.Possibility{
			Name: p.Name,
			Version: &dependency.VersionRelation{
				Number:   p.Version,
				Operator: "=",
			},
		})
		if pkg == nil {
			return nil, fmt.Errorf("could not find package %s in index", p.Name)
		}
		dfs.Push(&graphNode{
			pkg: pkg,
			r:   r,
		})
	}
	cb := func(n *graphNode) error {
		children, err := n.Children()
		if err != nil {
			return err
		}
		deps := make([]string, len(children))
		for i, c := range children {
			deps[i] = c.Name()
		}
		r.resolved[n.Name()] = r.indexPackageToPinned(n.pkg, deps)
		return nil
	}

	if err := dfs.PostOrderTraverse(cb); err != nil {
		return nil, err
	}

	out := []*spec.PinnedPackage{}
	for _, p := range r.resolved {
		out = append(out, p)
	}
	out = append(out, r.manualPkgs...)
	return out, nil
}

func (r *Resolver) packageFromRelation(rel dependency.Relation) (*index.Package, error) {
	for _, pos := range rel.Possibilities {
		if _, ok := r.excludes[pos.Name]; ok {
			// Consider an excluded package to have satisfied this relation.
			return nil, nil
		}
		p := r.packageFromPossibility(pos)
		if p == nil {
			// Skip any OR candidates that can't be resolved in the index.
			continue
		}
		// Greedily return the first OR candidate that can be resolved in the index.
		return p, nil
	}
	return nil, fmt.Errorf("cannot satisfy package relation %s", rel.String())
}

func (r *Resolver) packageFromPossibility(pos dependency.Possibility) *index.Package {
	options := []*index.PackageOption{}
	for _, ind := range r.indices {
		options = append(options, ind.PackageOptions(pos.Name)...)
	}
	pkg := selectOption(pos, options)
	return pkg
}

func (r *Resolver) indexPackageToPinned(p *index.Package, deps []string) *spec.PinnedPackage {
	prefix := strings.TrimSuffix(p.Repo.DownloadPrefix, "/")
	url := fmt.Sprintf("%s/%s", prefix, p.BinaryIndex.Filename)
	depPkgs := make([]*spec.PinnedPackage, 0)
	for _, name := range deps {
		resolved := r.resolved[name]
		// In the case of a cycle, some of the dependencies can be unresolved at this point.
		// For now, we just leave out those cyclic dependencies.
		// TODO(james): should figure out a better way to handle cycles.
		if resolved != nil {
			depPkgs = append(depPkgs, resolved)
		}
	}
	return &spec.PinnedPackage{
		Name:               p.BinaryIndex.Package,
		Version:            p.BinaryIndex.Version.String(),
		Arch:               p.Repo.Arch,
		URL:                url,
		SHA256:             p.BinaryIndex.SHA256,
		RepoName:           p.Repo.Name,
		DirectDependencies: depPkgs,
		ExcludePaths:       r.ps.ExcludePaths,
	}
}

func selectOption(pos dependency.Possibility, options []*index.PackageOption) *index.Package {
	// Sort first by whether the option is a virtual package and then by in descending version order.
	sort.Slice(options, func(i, j int) bool {
		if !options[i].Virtual && options[j].Virtual {
			return true
		}
		if options[i].Virtual && !options[j].Virtual {
			return false
		}
		return version.Compare(options[i].Pkg.BinaryIndex.Version, options[j].Pkg.BinaryIndex.Version) > 0
	})
	for _, opt := range options {
		p := opt.Pkg
		if pos.Version == nil || pos.Version.Number == "" || pos.Version.SatisfiedBy(p.BinaryIndex.Version) {
			return p
		}
	}
	// No valid options.
	return nil
}

// graphNode represents a node in the dependency graph for use with DFS.
// Since we don't want to enumerate the entire dependency graph of a debian package index,
// the graph node lazily determines its children.
// It implements Node from dfs.go.
type graphNode struct {
	pkg *index.Package
	r   *Resolver

	children []*graphNode
}

func (n *graphNode) Name() string {
	return n.pkg.BinaryIndex.Package
}

func (n *graphNode) Children() ([]*graphNode, error) {
	if n.children != nil {
		return n.children, nil
	}
	children := []*graphNode{}
	for _, rel := range n.pkg.BinaryIndex.GetDepends().Relations {
		p, err := n.r.packageFromRelation(rel)
		if err != nil {
			return nil, err
		}
		if p == nil {
			continue
		}
		child := &graphNode{
			pkg: p,
			r:   n.r,
		}
		n.children = append(n.children, child)
		children = append(children, child)
	}
	return children, nil
}
