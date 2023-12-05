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
