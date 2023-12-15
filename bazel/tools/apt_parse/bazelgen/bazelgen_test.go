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
					URL:                "amd64-download/pool/main/pkg1.deb",
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
					URL:                "amd64-download/pool/main/pkg1.deb",
					SHA256:             "3276ca056801384f1835efd3c4de88be3fb7e98b7de7ff59eb714126aaa287a7",
					RepoName:           "debian12",
					DirectDependencies: []string{"pkg2"},
				},
				{
					Name:               "pkg2",
					Version:            "0.2",
					Arch:               "x86_64",
					URL:                "amd64-download/pool/main/pkg2.deb",
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
						URL:                "amd64-download/pool/main/pkg1.deb",
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
						URL:                "amd64-download/pool/main/pkg1.deb",
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
						URL:                "amd64-download/pool/main/pkg1.deb",
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
					URL:                "amd64-download/pool/main/pkg1.deb",
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
					URL:                "amd64-download/pool/main/pkg1.deb",
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
						URL:                "amd64-download/pool/main/zzz.deb",
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
						URL:                "amd64-download/pool/main/aaa.deb",
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
