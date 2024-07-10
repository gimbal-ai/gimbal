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

package resolve_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/index"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/resolve"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec/testutils"
)

type fakeDownloader struct {
	indices map[string]string
}

func (d *fakeDownloader) Download(repo *spec.Repository) (*index.Index, error) {
	indexStr, ok := d.indices[repo.IndexURL]
	if !ok {
		return nil, fmt.Errorf("repo url %s not found in test case", repo.IndexURL)
	}
	r := strings.NewReader(indexStr)
	ind, err := index.NewIndex(r, repo)
	if err != nil {
		return nil, err
	}
	return ind, nil
}

func TestResolver(t *testing.T) {
	testCases := []struct {
		name           string
		arch           string
		indices        map[string]string
		packageSet     *spec.PackageSet
		expectedPinned []*testutils.PinnedPackage
	}{
		{
			name: "single dep no versions",
			arch: "amd64",
			indices: map[string]string{
				"amd64indexurl": `
Package: pkg1
Version: 0.1
Architecture: amd64
Depends: pkg2  (= 0.1)
Filename: pool/main/pkg1.deb
SHA256: e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c

Package: pkg2
Version: 0.1
Architecture: amd64
Depends: pkg3  (>> 0.1)
Filename: pool/main/pkg2.deb
SHA256: 96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7

Package: pkg3
Version: 0.2
Architecture: amd64
Filename: pool/main/pkg3.deb
SHA256: d07f3528c615545bb182269362926bc171adab565dade3e8c35cb3bae8ad9e5e
`,
			},
			packageSet: &spec.PackageSet{
				Repositories: []*spec.Repository{
					{
						Arch:           "amd64",
						IndexURL:       "amd64indexurl",
						DownloadPrefix: "amd64-download/",
					},
				},
				Packages: []*spec.Package{
					{
						Name: "pkg1",
					},
				},
			},
			expectedPinned: []*testutils.PinnedPackage{
				{
					Name:               "pkg1",
					Version:            "0.1",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c",
					DirectDependencies: []string{"pkg2"},
				},
				{
					Name:               "pkg2",
					Version:            "0.1",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg2.deb"},
					SHA256:             "96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7",
					DirectDependencies: []string{"pkg3"},
				},
				{
					Name:               "pkg3",
					Version:            "0.2",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg3.deb"},
					SHA256:             "d07f3528c615545bb182269362926bc171adab565dade3e8c35cb3bae8ad9e5e",
					DirectDependencies: []string{},
				},
			},
		},
		{
			name: "multiple version options",
			arch: "amd64",
			indices: map[string]string{
				"amd64indexurl": `
Package: pkg1
Version: 0.1
Architecture: amd64
Depends: pkg2  (= 0.1)
Filename: pool/main/pkg1.deb
SHA256: e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c

Package: pkg1
Version: 0.2
Architecture: amd64
Depends: pkg2  (= 0.2)
Filename: pool/main/pkg1.deb
SHA256: 0854f5fd221a7caf387bf519abd3937491b6dc7f8921e63ef8a796e5ef684f3c

Package: pkg2
Version: 0.1
Architecture: amd64
Depends: pkg3  (>> 0.1)
Filename: pool/main/pkg2.deb
SHA256: 96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7

Package: pkg2
Version: 0.2
Architecture: amd64
Depends: pkg3  (>> 0.1)
Filename: pool/main/pkg2.deb
SHA256: eefa940924affb4bfba49fddbff9080156c01380db3b162cd423c6ab1b31b5b8

Package: pkg3
Version: 0.2
Architecture: amd64
Filename: pool/main/pkg3.deb
SHA256: d07f3528c615545bb182269362926bc171adab565dade3e8c35cb3bae8ad9e5e
`,
			},
			packageSet: &spec.PackageSet{
				Repositories: []*spec.Repository{
					{
						Arch:           "amd64",
						IndexURL:       "amd64indexurl",
						DownloadPrefix: "amd64-download/",
					},
				},
				Packages: []*spec.Package{
					{
						Name:    "pkg1",
						Version: "0.2",
					},
				},
			},
			expectedPinned: []*testutils.PinnedPackage{
				{
					Name:               "pkg1",
					Version:            "0.2",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "0854f5fd221a7caf387bf519abd3937491b6dc7f8921e63ef8a796e5ef684f3c",
					DirectDependencies: []string{"pkg2"},
				},
				{
					Name:               "pkg2",
					Version:            "0.2",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg2.deb"},
					SHA256:             "eefa940924affb4bfba49fddbff9080156c01380db3b162cd423c6ab1b31b5b8",
					DirectDependencies: []string{"pkg3"},
				},
				{
					Name:               "pkg3",
					Version:            "0.2",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg3.deb"},
					SHA256:             "d07f3528c615545bb182269362926bc171adab565dade3e8c35cb3bae8ad9e5e",
					DirectDependencies: []string{},
				},
			},
		},
		{
			name: "or in depends",
			arch: "amd64",
			indices: map[string]string{
				"amd64indexurl": `
Package: pkg1
Version: 0.1
Architecture: amd64
Depends: pkg3 (<< 0.2) | pkg2  (= 0.1)
Filename: pool/main/pkg1.deb
SHA256: e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c

Package: pkg2
Version: 0.1
Architecture: amd64
Depends: pkg3  (>> 0.1)
Filename: pool/main/pkg2.deb
SHA256: 96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7

Package: pkg3
Version: 0.2
Architecture: amd64
Filename: pool/main/pkg3.deb
SHA256: d07f3528c615545bb182269362926bc171adab565dade3e8c35cb3bae8ad9e5e
`,
			},
			packageSet: &spec.PackageSet{
				Repositories: []*spec.Repository{
					{
						Arch:           "amd64",
						IndexURL:       "amd64indexurl",
						DownloadPrefix: "amd64-download/",
					},
				},
				Packages: []*spec.Package{
					{
						Name:    "pkg1",
						Version: "0.1",
					},
				},
			},
			expectedPinned: []*testutils.PinnedPackage{
				{
					Name:               "pkg1",
					Version:            "0.1",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c",
					DirectDependencies: []string{"pkg2"},
				},
				{
					Name:               "pkg2",
					Version:            "0.1",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg2.deb"},
					SHA256:             "96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7",
					DirectDependencies: []string{"pkg3"},
				},
				{
					Name:               "pkg3",
					Version:            "0.2",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg3.deb"},
					SHA256:             "d07f3528c615545bb182269362926bc171adab565dade3e8c35cb3bae8ad9e5e",
					DirectDependencies: []string{},
				},
			},
		},
		{
			name: "virtual package",
			arch: "amd64",
			indices: map[string]string{
				"amd64indexurl": `
Package: pkg1
Version: 0.1
Architecture: amd64
Depends: pkg3 (<< 0.2) | pkg2  (= 0.1)
Filename: pool/main/pkg1.deb
SHA256: e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c

Package: pkg2
Version: 0.1
Architecture: amd64
Depends: pkg3  (>> 0.1), virtualpkg (= 0.2)
Filename: pool/main/pkg2.deb
SHA256: 96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7

Package: pkg3
Version: 0.2
Architecture: amd64
Filename: pool/main/pkg3.deb
SHA256: d07f3528c615545bb182269362926bc171adab565dade3e8c35cb3bae8ad9e5e

Package: pkg4
Version: 0.2
Architecture: amd64
Filename: pool/main/pkg4.deb
Provides: virtualpkg
SHA256: 3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7
`,
			},
			packageSet: &spec.PackageSet{
				Repositories: []*spec.Repository{
					{
						Arch:           "amd64",
						IndexURL:       "amd64indexurl",
						DownloadPrefix: "amd64-download/",
					},
				},
				Packages: []*spec.Package{
					{
						Name:    "pkg1",
						Version: "0.1",
					},
				},
			},
			expectedPinned: []*testutils.PinnedPackage{
				{
					Name:               "pkg1",
					Version:            "0.1",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c",
					DirectDependencies: []string{"pkg2"},
				},
				{
					Name:               "pkg2",
					Version:            "0.1",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg2.deb"},
					SHA256:             "96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7",
					DirectDependencies: []string{"pkg3", "pkg4"},
				},
				{
					Name:               "pkg3",
					Version:            "0.2",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg3.deb"},
					SHA256:             "d07f3528c615545bb182269362926bc171adab565dade3e8c35cb3bae8ad9e5e",
					DirectDependencies: []string{},
				},
				{
					Name:               "pkg4",
					Version:            "0.2",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg4.deb"},
					SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
					DirectDependencies: []string{},
				},
			},
		},
		{
			name: "cyclic deps",
			arch: "amd64",
			indices: map[string]string{
				"amd64indexurl": `
Package: pkg1
Version: 0.1
Architecture: amd64
Depends: pkg2  (= 0.1)
Filename: pool/main/pkg1.deb
SHA256: e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c

Package: pkg2
Version: 0.1
Architecture: amd64
Depends: pkg1  (= 0.1)
Filename: pool/main/pkg2.deb
SHA256: 96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7
`,
			},
			packageSet: &spec.PackageSet{
				Repositories: []*spec.Repository{
					{
						Arch:           "amd64",
						IndexURL:       "amd64indexurl",
						DownloadPrefix: "amd64-download/",
					},
				},
				Packages: []*spec.Package{
					{
						Name: "pkg1",
					},
				},
			},
			expectedPinned: []*testutils.PinnedPackage{
				{
					Name:               "pkg1",
					Version:            "0.1",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c",
					DirectDependencies: []string{"pkg2"},
				},
				{
					Name:    "pkg2",
					Version: "0.1",
					Arch:    "amd64",
					URLs:    []string{"amd64-download/pool/main/pkg2.deb"},
					SHA256:  "96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7",
					// The pkg that is visited first won't declare the cyclic dependency.
					DirectDependencies: []string{},
				},
			},
		},
		{
			name: "exclude packages",
			arch: "amd64",
			indices: map[string]string{
				"amd64indexurl": `
Package: pkg1
Version: 0.1
Architecture: amd64
Depends: pkg2  (= 0.1)
Filename: pool/main/pkg1.deb
SHA256: e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c

Package: pkg2
Version: 0.1
Architecture: amd64
Depends: pkg3  (>> 0.1)
Filename: pool/main/pkg2.deb
SHA256: 96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7

Package: pkg3
Version: 0.2
Architecture: amd64
Filename: pool/main/pkg3.deb
SHA256: d07f3528c615545bb182269362926bc171adab565dade3e8c35cb3bae8ad9e5e
`,
			},
			packageSet: &spec.PackageSet{
				Repositories: []*spec.Repository{
					{
						Arch:           "amd64",
						IndexURL:       "amd64indexurl",
						DownloadPrefix: "amd64-download/",
					},
				},
				Packages: []*spec.Package{
					{
						Name: "pkg1",
					},
				},
				ExcludePackages: []string{
					"pkg3",
				},
			},
			expectedPinned: []*testutils.PinnedPackage{
				{
					Name:               "pkg1",
					Version:            "0.1",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "e410f3d2da35bccf757976d38e6a309d4d94d25c0dfde565a9662e3f75951b2c",
					DirectDependencies: []string{"pkg2"},
				},
				{
					Name:               "pkg2",
					Version:            "0.1",
					Arch:               "amd64",
					URLs:               []string{"amd64-download/pool/main/pkg2.deb"},
					SHA256:             "96c2d796a21fdc92b4d272a550841c208e89c91ab0d54514ac28ae92da64c2c7",
					DirectDependencies: []string{},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			d := &fakeDownloader{
				indices: tc.indices,
			}
			r, err := resolve.NewResolver(d, tc.packageSet, tc.arch)
			require.NoError(t, err)
			actual, err := r.Resolve()
			require.NoError(t, err)
			expected, err := testutils.ConvertToSpecPackages(tc.expectedPinned)
			require.NoError(t, err)
			assert.ElementsMatch(t, expected, actual)
		})
	}
}
