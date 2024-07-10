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

package index_test

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/index"
)

type pkgOpt struct {
	name    string
	version string
	virtual bool
}

func testPkgOptFromIndexPkgOpt(p *index.PackageOption) *pkgOpt {
	return &pkgOpt{
		name:    p.Pkg.BinaryIndex.Package,
		version: p.Pkg.BinaryIndex.Version.String(),
		virtual: p.Virtual,
	}
}

func TestIndex(t *testing.T) {
	testCases := []struct {
		name            string
		indexFile       string
		expectedOptions map[string][]*pkgOpt
	}{
		{
			name: "simple package parse",
			indexFile: `
Package: pkg1
Version: 0.1`,
			expectedOptions: map[string][]*pkgOpt{
				"pkg1": {
					{
						name:    "pkg1",
						version: "0.1",
						virtual: false,
					},
				},
			},
		},
		{
			name: "single dep",
			indexFile: `
Package: pkg1
Version: 0.1
Depends: pkg2

Package: pkg2
Version: 0.2`,
			expectedOptions: map[string][]*pkgOpt{
				"pkg1": {
					{
						name:    "pkg1",
						version: "0.1",
						virtual: false,
					},
				},
				"pkg2": {
					{
						name:    "pkg2",
						version: "0.2",
						virtual: false,
					},
				},
			},
		},
		{
			name: "virtual pkg",
			indexFile: `
Package: pkg1
Version: 0.1
Provides: virt

Package: pkg2
Version: 0.2
Provides: virt

Package: virt
Version: 0.0.1`,
			expectedOptions: map[string][]*pkgOpt{
				"virt": {
					{
						name:    "pkg1",
						version: "0.1",
						virtual: true,
					},
					{
						name:    "pkg2",
						version: "0.2",
						virtual: true,
					},
					{
						name:    "virt",
						version: "0.0.1",
						virtual: false,
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			r := strings.NewReader(tc.indexFile)
			ind, err := index.NewIndex(r, nil)
			require.NoError(t, err)
			for name, expectedOpts := range tc.expectedOptions {
				pkgOpts := ind.PackageOptions(name)
				actualOpts := []*pkgOpt{}
				for _, po := range pkgOpts {
					actualOpts = append(actualOpts, testPkgOptFromIndexPkgOpt(po))
				}

				assert.ElementsMatch(t, expectedOpts, actualOpts)
			}
		})
	}
}
