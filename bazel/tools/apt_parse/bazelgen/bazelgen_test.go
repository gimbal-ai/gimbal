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

package bazelgen_test

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/bazelgen"
	"gimletlabs.ai/gimlet/bazel/tools/apt_parse/spec/testutils"
)

func TestBazelGenerator(t *testing.T) {
	testCases := []struct {
		name        string
		pkgSets     [][]*testutils.PinnedPackage
		macroName   string
		expectedBzl string
	}{
		{
			name:      "single package",
			macroName: "load_debs",
			pkgSets: [][]*testutils.PinnedPackage{{
				{
					// Make sure "+" gets sanitized correctly.
					Name:               "pkg1++",
					Version:            "0.1",
					Arch:               "x86_64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
					RepoName:           "debian12",
					DirectDependencies: []string{},
				},
			}},
			expectedBzl: `load("//bazel/rules_pkg:pkg_provider_archives.bzl", "deb_archive_w_pkg_providers")

def load_debs():
    deb_archive_w_pkg_providers(
        name = "debian12_pkg1___x86_64",
        sha256 = "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
        urls = ["amd64-download/pool/main/pkg1.deb"],
        deps = [],
    )
`,
		},
		{
			name:      "with dep",
			macroName: "load_debs",
			pkgSets: [][]*testutils.PinnedPackage{{
				{
					Name:               "pkg1",
					Version:            "0.1",
					Arch:               "x86_64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
					RepoName:           "debian12",
					DirectDependencies: []string{"pkg2"},
				},
				{
					Name:               "pkg2",
					Version:            "0.2",
					Arch:               "x86_64",
					URLs:               []string{"amd64-download/pool/main/pkg2.deb"},
					SHA256:             "59bff2abde5ab0e575200bfebf64dbaef65e7f9361fa1ad445bfde19f1372a8e",
					RepoName:           "debian11",
					DirectDependencies: []string{},
				},
			}},
			expectedBzl: `load("//bazel/rules_pkg:pkg_provider_archives.bzl", "deb_archive_w_pkg_providers")

def load_debs():
    deb_archive_w_pkg_providers(
        name = "debian11_pkg2_x86_64",
        sha256 = "59bff2abde5ab0e575200bfebf64dbaef65e7f9361fa1ad445bfde19f1372a8e",
        urls = ["amd64-download/pool/main/pkg2.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_pkg1_x86_64",
        sha256 = "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
        urls = ["amd64-download/pool/main/pkg1.deb"],
        deps = ["@debian11_pkg2_x86_64//:all_files"],
    )
`,
		},
		{
			name:      "duplicates",
			macroName: "load_debs",
			pkgSets: [][]*testutils.PinnedPackage{
				{
					{
						Name:               "pkg1",
						Version:            "0.1",
						Arch:               "x86_64",
						URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
						SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
						RepoName:           "debian12",
						DirectDependencies: []string{},
					},
				},
				{
					// This package has a different RepoName so it should still be included.
					{
						Name:               "pkg1",
						Version:            "0.1",
						Arch:               "x86_64",
						URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
						SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
						RepoName:           "debian11",
						DirectDependencies: []string{},
					},
				},
				{
					// This package has the same name version and repo as the first package so it shouldn't be added to the bzl file.
					{
						Name:               "pkg1",
						Version:            "0.1",
						Arch:               "x86_64",
						URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
						SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
						RepoName:           "debian12",
						DirectDependencies: []string{},
					},
				},
			},
			expectedBzl: `load("//bazel/rules_pkg:pkg_provider_archives.bzl", "deb_archive_w_pkg_providers")

def load_debs():
    deb_archive_w_pkg_providers(
        name = "debian11_pkg1_x86_64",
        sha256 = "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
        urls = ["amd64-download/pool/main/pkg1.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_pkg1_x86_64",
        sha256 = "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
        urls = ["amd64-download/pool/main/pkg1.deb"],
        deps = [],
    )
`,
		},
		{
			name:      "exclude paths",
			macroName: "load_debs",
			pkgSets: [][]*testutils.PinnedPackage{{
				{
					Name:               "pkg1",
					Version:            "0.1",
					Arch:               "x86_64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
					RepoName:           "debian12",
					DirectDependencies: []string{},
					ExcludePaths:       []string{"a/b/c", "b/c/d"},
				},
			}},
			expectedBzl: `load("//bazel/rules_pkg:pkg_provider_archives.bzl", "deb_archive_w_pkg_providers")

def load_debs():
    deb_archive_w_pkg_providers(
        name = "debian12_pkg1_x86_64",
        exclude_paths = ["a/b/c", "b/c/d"],
        sha256 = "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
        urls = ["amd64-download/pool/main/pkg1.deb"],
        deps = [],
    )
`,
		},
		{
			name:      "extra symlinks",
			macroName: "load_debs",
			pkgSets: [][]*testutils.PinnedPackage{{
				{
					Name:               "pkg1",
					Version:            "0.1",
					Arch:               "x86_64",
					URLs:               []string{"amd64-download/pool/main/pkg1.deb"},
					SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
					RepoName:           "debian12",
					DirectDependencies: []string{},
					ExcludePaths:       []string{},
					ExtraSymlinks:      map[string]string{"/etc/a": "/usr/sbin/a"},
				},
			}},
			expectedBzl: `load("//bazel/rules_pkg:pkg_provider_archives.bzl", "deb_archive_w_pkg_providers")

def load_debs():
    deb_archive_w_pkg_providers(
        name = "debian12_pkg1_x86_64",
        extra_symlinks = {
            "/etc/a": "/usr/sbin/a",
        },
        sha256 = "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
        urls = ["amd64-download/pool/main/pkg1.deb"],
        deps = [],
    )
`,
		},
		{
			name:      "sorted by package name",
			macroName: "load_debs",
			pkgSets: [][]*testutils.PinnedPackage{
				{
					{
						Name:               "zzz",
						Version:            "0.1",
						Arch:               "x86_64",
						URLs:               []string{"amd64-download/pool/main/zzz.deb"},
						SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
						RepoName:           "debian12",
						DirectDependencies: []string{},
					},
				},
				{
					{
						Name:               "aaa",
						Version:            "0.1",
						Arch:               "x86_64",
						URLs:               []string{"amd64-download/pool/main/aaa.deb"},
						SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
						RepoName:           "debian12",
						DirectDependencies: []string{},
					},
				},
			},
			expectedBzl: `load("//bazel/rules_pkg:pkg_provider_archives.bzl", "deb_archive_w_pkg_providers")

def load_debs():
    deb_archive_w_pkg_providers(
        name = "debian12_aaa_x86_64",
        sha256 = "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
        urls = ["amd64-download/pool/main/aaa.deb"],
        deps = [],
    )
    deb_archive_w_pkg_providers(
        name = "debian12_zzz_x86_64",
        sha256 = "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
        urls = ["amd64-download/pool/main/zzz.deb"],
        deps = [],
    )
`,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			gen, err := bazelgen.NewBazelGenerator(tc.macroName)
			require.NoError(t, err)
			for _, set := range tc.pkgSets {
				pkgs, err := testutils.ConvertToSpecPackages(set)
				require.NoError(t, err)
				gen.AddPinnedSet(pkgs)
			}
			assert.Equal(t, tc.expectedBzl, gen.Content())
		})
	}
}
