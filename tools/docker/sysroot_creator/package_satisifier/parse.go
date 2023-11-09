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
	"bufio"
	"fmt"
	"io"
	"regexp"
	"strings"

	debVersion "pault.ag/go/debian/version"
)

var singleDepRegex *regexp.Regexp

func init() {
	depRegex := `(?P<pkg_name>[^ :]*)(?P<arch>[:][^ ]+)?( \((?P<version_op>[<>=]+) (?P<version>[^)]+)\))?`
	singleDepRegex = regexp.MustCompile(depRegex)
}

type versionCompareType string

const (
	lessThan      versionCompareType = "<<"
	lessThanEq    versionCompareType = "<="
	eq            versionCompareType = "="
	greaterThanEq versionCompareType = ">="
	greaterThan   versionCompareType = ">>"
)

type versionedDependency struct {
	name       string
	version    *debVersion.Version
	versionCmp versionCompareType
}

func (d *versionedDependency) versionString() string {
	if d.version == nil {
		return ""
	}
	return d.version.String()
}

func (d *versionedDependency) String() string {
	return fmt.Sprintf("%s (%s %s)", d.name, d.versionCmp, d.versionString())
}

type dependency struct {
	// anyOf stores each package that could satisfy this dependency. i.e. an OR dependency on each of the packages listed.
	anyOf []*versionedDependency
}

func (d *dependency) String() string {
	strs := []string{}
	for _, vd := range d.anyOf {
		strs = append(strs, vd.String())
	}
	return strings.Join(strs, " | ")
}

type pkg struct {
	name     string
	version  debVersion.Version
	filename string
	depends  []*dependency
	// list of virtual packages provided by this package.
	provides       []*versionedDependency
	downloadPrefix string
}

func (p *pkg) versionString() string {
	return p.version.String()
}

type (
	pkgVersions        map[string]*pkg
	versionedPkgSet    map[string]pkgVersions
	versionedProviders map[string][]*versionedDependency
)

type database struct {
	packages         versionedPkgSet
	virtualProviders map[string]versionedProviders
}

func (p *pkg) String() string {
	b := &strings.Builder{}
	_, _ = b.WriteString("Package: " + p.name + "\n")
	_, _ = b.WriteString("Version: " + p.versionString() + "\n")
	_, _ = b.WriteString("Depends:\n")
	for _, dep := range p.depends {
		_, _ = b.WriteString(dep.String())
		_, _ = b.WriteString("\n")
	}
	_, _ = b.WriteString("Provides:\n")
	for _, dep := range p.provides {
		_, _ = b.WriteString(dep.String() + "\n")
	}
	return b.String()
}

func newPkg(name string, downloadPrefix string) *pkg {
	return &pkg{
		name:           name,
		depends:        make([]*dependency, 0),
		provides:       make([]*versionedDependency, 0),
		downloadPrefix: downloadPrefix,
	}
}

func parseProvides(str string, curPkg *pkg) error {
	provides := strings.Split(str, ", ")
	for _, depStr := range provides {
		dep, err := parseVersionedDep(depStr)
		if err != nil {
			return err
		}
		curPkg.provides = append(curPkg.provides, dep)
	}
	return nil
}

func parseVersionedDep(depStr string) (*versionedDependency, error) {
	matches := singleDepRegex.FindStringSubmatch(depStr)
	result := make(map[string]string)
	for i, name := range singleDepRegex.SubexpNames() {
		if i != 0 && name != "" {
			result[name] = matches[i]
		}
	}
	if result["pkg_name"] == "" {
		return nil, fmt.Errorf("could not parse dependency: %s", depStr)
	}

	dep := &versionedDependency{
		name: result["pkg_name"],
	}

	if result["version_op"] != "" {
		dep.versionCmp = versionCompareType(result["version_op"])
	}
	if result["version"] != "" {
		ver, err := debVersion.Parse(result["version"])
		if err != nil {
			return nil, err
		}
		dep.version = &ver
	}

	return dep, nil
}

func parseDep(str string) (*dependency, error) {
	depStrings := strings.Split(str, " | ")
	dep := &dependency{
		anyOf: make([]*versionedDependency, 0),
	}
	for _, depStr := range depStrings {
		depPkg, err := parseVersionedDep(depStr)
		if err != nil {
			return nil, err
		}
		dep.anyOf = append(dep.anyOf, depPkg)
	}
	return dep, nil
}

func parseDepends(str string, curPkg *pkg) error {
	depends := strings.Split(str, ", ")
	for _, depStr := range depends {
		dep, err := parseDep(depStr)
		if err != nil {
			return err
		}
		curPkg.depends = append(curPkg.depends, dep)
	}
	return nil
}

func newPackageDatabase() *database {
	return &database{
		packages:         make(versionedPkgSet),
		virtualProviders: make(map[string]versionedProviders),
	}
}

func (db *database) addPackage(p *pkg) {
	if _, ok := db.packages[p.name]; !ok {
		db.packages[p.name] = make(pkgVersions)
	}
	versions := db.packages[p.name]
	versions[p.versionString()] = p
}

func (db *database) addVirtualPkgWithProvider(virtual *versionedDependency, provider *pkg) {
	if _, ok := db.virtualProviders[virtual.name]; !ok {
		db.virtualProviders[virtual.name] = make(versionedProviders)
	}
	vp := db.virtualProviders[virtual.name]
	if _, ok := vp[virtual.versionString()]; !ok {
		vp[virtual.versionString()] = make([]*versionedDependency, 0, 1)
	}
	vp[virtual.versionString()] = append(vp[virtual.versionString()], &versionedDependency{
		name:       provider.name,
		version:    &provider.version,
		versionCmp: eq,
	})
}

func (db *database) parsePackageDatabase(r io.Reader, downloadPrefix string) error {
	var curPkg *pkg
	scanner := bufio.NewScanner(r)
	// Increase maximum token size of scanner.
	bufSize := 512 * 1024
	buf := make([]byte, 0, bufSize)
	scanner.Buffer(buf, bufSize)
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "Package:") {
			if curPkg != nil {
				db.addPackage(curPkg)
			}
			curPkg = newPkg(strings.TrimPrefix(line, "Package: "), downloadPrefix)
		}
		if strings.HasPrefix(line, "Version: ") {
			ver, err := debVersion.Parse(strings.TrimPrefix(line, "Version: "))
			if err != nil {
				return err
			}
			curPkg.version = ver
		}
		if strings.HasPrefix(line, "Provides: ") {
			if err := parseProvides(strings.TrimPrefix(line, "Provides: "), curPkg); err != nil {
				return err
			}

			for _, virtualPkg := range curPkg.provides {
				db.addVirtualPkgWithProvider(virtualPkg, curPkg)
			}
		}
		if strings.HasPrefix(line, "Pre-Depends: ") {
			if err := parseDepends(strings.TrimPrefix(line, "Pre-Depends: "), curPkg); err != nil {
				return err
			}
		}
		if strings.HasPrefix(line, "Depends: ") {
			if err := parseDepends(strings.TrimPrefix(line, "Depends: "), curPkg); err != nil {
				return err
			}
		}
		if strings.HasPrefix(line, "Filename: ") {
			curPkg.filename = strings.TrimPrefix(line, "Filename: ")
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	if curPkg != nil {
		db.addPackage(curPkg)
	}
	return nil
}
