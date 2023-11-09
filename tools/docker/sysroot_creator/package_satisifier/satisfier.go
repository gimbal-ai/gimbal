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
	"fmt"
	"sort"

	debVersion "pault.ag/go/debian/version"
)

type depSatisfier struct {
	db       *database
	excludes map[string]bool
	includes []string

	visited map[string]bool
}

func newDepSatisfier(db *database, spec *spec) *depSatisfier {
	ds := &depSatisfier{
		db:       db,
		excludes: make(map[string]bool),
		includes: spec.Includes,

		visited: make(map[string]bool),
	}
	for _, exc := range spec.Excludes {
		ds.excludes[exc] = true
	}
	return ds
}

func (vd *versionedDependency) selectVersion(versions []debVersion.Version) (*debVersion.Version, error) {
	sort.Sort(debVersion.Slice(versions))
	if vd.version == nil {
		return &versions[len(versions)-1], nil
	}
	if vd.versionCmp == greaterThan {
		ver := versions[len(versions)-1]
		if debVersion.Compare(ver, *vd.version) > 0 {
			return &ver, nil
		}
		return nil, fmt.Errorf("no versions >> %s for %s, closest %s, %v", vd.version, vd.name, ver, versions)
	}
	if vd.versionCmp == greaterThanEq {
		ver := versions[len(versions)-1]
		if debVersion.Compare(ver, *vd.version) >= 0 {
			return &ver, nil
		}
		return nil, fmt.Errorf("no versions >= %s for %s, closest %s, all %v", vd.version, vd.name, ver, versions)
	}
	if vd.versionCmp == lessThan {
		firstGreaterEq := sort.Search(len(versions), func(i int) bool {
			return debVersion.Compare(versions[i], *vd.version) >= 0
		})
		if firstGreaterEq == 0 {
			return nil, fmt.Errorf("no version << %s for %s, closest %s, all %v", vd.version, vd.name, versions[firstGreaterEq], versions)
		}
		return &versions[firstGreaterEq-1], nil
	}
	if vd.versionCmp == lessThanEq {
		firstGreater := sort.Search(len(versions), func(i int) bool {
			return debVersion.Compare(versions[i], *vd.version) > 0
		})
		if firstGreater == 0 {
			return nil, fmt.Errorf("no version << %s for %s, closest %s, all %v", vd.version, vd.name, versions[firstGreater], versions)
		}
		return &versions[firstGreater-1], nil
	}
	if vd.versionCmp == eq {
		index := sort.Search(len(versions), func(i int) bool {
			return debVersion.Compare(versions[i], *vd.version) >= 0
		})
		if index == len(versions) || debVersion.Compare(versions[index], *vd.version) != 0 {
			return nil, fmt.Errorf("no version == %s for %s, all %v", vd.version, vd.name, versions)
		}
		return &versions[index], nil
	}
	return nil, fmt.Errorf("unrecognized version operator %s", vd.versionCmp)
}

func (ds *depSatisfier) getPackage(dep *versionedDependency) (*pkg, error) {
	versionToPkg, ok := ds.db.packages[dep.name]
	if !ok {
		return nil, nil
	}
	versions := make([]debVersion.Version, 0, len(versionToPkg))
	for _, pkg := range versionToPkg {
		versions = append(versions, pkg.version)
	}
	if len(versions) == 0 {
		return nil, nil
	}
	ver, err := dep.selectVersion(versions)
	if err != nil {
		return nil, err
	}
	return versionToPkg[ver.String()], nil
}

func (ds *depSatisfier) listRequiredDebs() ([]string, error) {
	// We do a breadth first search of the package dependency tree, to find all dependencies of the specified includes.
	q := make([]*versionedDependency, 0, len(ds.includes))
	for _, depStr := range ds.includes {
		dep, err := parseVersionedDep(depStr)
		if err != nil {
			return nil, err
		}
		q = append(q, dep)
	}

	debs := make([]string, 0)
	includedPackages := make(map[string]string)

	for len(q) > 0 {
		dep := q[0]
		q = q[1:]
		// Don't follow dependencies of excluded packages.
		if ds.excludes[dep.name] {
			continue
		}

		p, err := ds.getPackage(dep)
		if err != nil {
			return nil, err
		}
		// If p is nil, then this package name refers to a virtual package.
		if p == nil {
			// If a virtual package is required. We need to choose a provider for it.
			provider, err := ds.findProvider(dep)
			if err != nil {
				return nil, err
			}
			if provider != nil {
				q = append(q, provider)
			}
			continue
		}
		if ds.visited[p.name+";"+p.versionString()] {
			continue
		}
		ds.visited[p.name+";"+p.versionString()] = true
		ds.visited[p.name] = true

		if otherVersion, ok := includedPackages[p.name]; ok {
			return nil, fmt.Errorf(
				"attempting to include two versions of the same package (%s vs %s):%s",
				otherVersion,
				p.versionString(),
				p.String(),
			)
		}
		url := fmt.Sprintf("%s/%s", p.downloadPrefix, p.filename)
		debs = append(debs, url)
		includedPackages[p.name] = p.versionString()

		// Mark all virtual packages this real package provides as visited.
		for _, provided := range p.provides {
			ds.visited[provided.name+";"+provided.versionString()] = true
			ds.visited[provided.name] = true
		}
		for _, d := range p.depends {
			dep := ds.resolveDep(d)
			if dep != nil {
				q = append(q, dep)
			}
		}
	}
	return debs, nil
}

func (ds *depSatisfier) resolveDep(d *dependency) *versionedDependency {
	if len(d.anyOf) == 0 {
		return nil
	}
	if len(d.anyOf) == 1 {
		return d.anyOf[0]
	}
	// At this point, the dependency contains an OR.
	// We first try to see if any of the OR dependencies have already been satisified.
	// Failing that we naively pick the first one of the OR that exists in the package database.
	// We could keep the OR condition around until the very end and
	// then try to satisfy all the ORs minimally using a SAT solver or something of the like,
	// but that's too much effort for now.

	candidates := []*versionedDependency{}
	for _, vd := range d.anyOf {
		if (vd.versionString() == "" && ds.visited[vd.name]) || ds.visited[vd.name+";"+vd.versionString()] {
			// Since its already visited we don't need to traverse it.
			return nil
		}
		// Choose the first dependency that exists in the package database.
		p, err := ds.getPackage(vd)
		if err != nil {
			continue
		}
		if p == nil {
			_, err := ds.findProvider(vd)
			if err != nil {
				continue
			}
		}
		candidates = append(candidates, vd)
	}

	return candidates[0]
}

func (ds *depSatisfier) findProvider(dep *versionedDependency) (*versionedDependency, error) {
	vp, ok := ds.db.virtualProviders[dep.name]
	if !ok || len(vp) == 0 {
		return nil, fmt.Errorf("cannot find provider for virtual package '%s'", dep.name)
	}

	versions := make([]debVersion.Version, 0, len(vp))
	for verStr := range vp {
		if verStr != "" {
			ver, err := debVersion.Parse(verStr)
			if err != nil {
				return nil, err
			}
			versions = append(versions, ver)
		}
	}

	var pkgSet []*versionedDependency
	if len(versions) == 0 {
		pkgSet = vp[""]
	} else {
		ver, err := dep.selectVersion(versions)
		if err != nil {
			return nil, err
		}
		pkgSet = vp[ver.String()]
	}

	var first *versionedDependency
	for _, p := range pkgSet {
		first = p
		if ds.visited[p.name+";"+p.versionString()] {
			return nil, nil
		}
	}

	// If we haven't already visited one of the providers, then choose the first one.
	return first, nil
}
