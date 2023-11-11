# Copyright © 2023- Gimlet Labs, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Gimlet Labs, Inc. and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Gimlet Labs, Inc. and its suppliers and
# may be covered by U.S. and Foreign Patents, patents in process,
# and are protected by trade secret or copyright law. Dissemination
# of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from
# Gimlet Labs, Inc.
#
# SPDX-License-Identifier: Proprietary

# THIS FILE IS GENERATED BY build_all_sysroots

load("//bazel/cc_toolchains/sysroots:sysroots.bzl", "sysroot_repo")

def _create_sysroots():
    sysroot_repo(
        name = "sysroot_aarch64_glibc2_36_runtime",
        target_arch = "aarch64",
        variant = "runtime",
        libc_version = "glibc2_36",
        sha256 = "e5b870e5cdad7a9a441bd9f668d02478bbd83ee870a3cfeca8e325b109808f89",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-aarch64-glibc2_36-runtime.tar.gz"],
        sysroot_features = [],
        disabled_for_features = ["jetson"],
    )
    sysroot_repo(
        name = "sysroot_aarch64_glibc2_36_build",
        target_arch = "aarch64",
        variant = "build",
        libc_version = "glibc2_36",
        sha256 = "cf08dbeaea9726e40373dd93d2d398c333ba9eec938973c2c7871cad7a32c62f",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-aarch64-glibc2_36-build.tar.gz"],
        sysroot_features = [],
        disabled_for_features = ["jetson"],
    )
    sysroot_repo(
        name = "sysroot_aarch64_glibc2_36_test",
        target_arch = "aarch64",
        variant = "test",
        libc_version = "glibc2_36",
        sha256 = "de1a29f5c1e56f54f8f20587ada51a9859b680fbe5770688961101ad91a752c9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-aarch64-glibc2_36-test.tar.gz"],
        sysroot_features = [],
        disabled_for_features = ["debug", "jetson"],
    )
    sysroot_repo(
        name = "sysroot_aarch64_glibc2_36_test_debug",
        target_arch = "aarch64",
        variant = "test",
        libc_version = "glibc2_36",
        sha256 = "c3393c321a2889e61d21c0e3f20cc441bd32b8bc78224fae9200158f1e2a36fb",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-aarch64-glibc2_36-test-debug.tar.gz"],
        sysroot_features = ["debug"],
        disabled_for_features = ["jetson"],
    )
    sysroot_repo(
        name = "sysroot_x86_64_glibc2_36_runtime",
        target_arch = "x86_64",
        variant = "runtime",
        libc_version = "glibc2_36",
        sha256 = "6eba5bef6d2078079ed2ca0c7d64d122556432e1027867b0104937b4218f9e91",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-x86_64-glibc2_36-runtime.tar.gz"],
        sysroot_features = [],
        disabled_for_features = ["jetson"],
    )
    sysroot_repo(
        name = "sysroot_x86_64_glibc2_36_build",
        target_arch = "x86_64",
        variant = "build",
        libc_version = "glibc2_36",
        sha256 = "63f0f32135e0e6d13bb7281c55aa9bdcfc818c70b805d28cafa85b9a247a3e28",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-x86_64-glibc2_36-build.tar.gz"],
        sysroot_features = [],
        disabled_for_features = ["jetson"],
    )
    sysroot_repo(
        name = "sysroot_x86_64_glibc2_36_test",
        target_arch = "x86_64",
        variant = "test",
        libc_version = "glibc2_36",
        sha256 = "cb737ffd794ce0dc187ab6c8ec432a6bdd7500dec1138fa89036fdcb6d0e49e9",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-x86_64-glibc2_36-test.tar.gz"],
        sysroot_features = [],
        disabled_for_features = ["debug", "jetson"],
    )
    sysroot_repo(
        name = "sysroot_x86_64_glibc2_36_test_debug",
        target_arch = "x86_64",
        variant = "test",
        libc_version = "glibc2_36",
        sha256 = "d79f53655057ea5770c75c2afab3816b7a990693468fdbb34f47f36e6130bae6",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-x86_64-glibc2_36-test-debug.tar.gz"],
        sysroot_features = ["debug"],
        disabled_for_features = ["jetson"],
    )
    sysroot_repo(
        name = "sysroot_aarch64_glibc2_31_runtime_jetson",
        target_arch = "aarch64",
        variant = "runtime",
        libc_version = "glibc2_31",
        sha256 = "0677155de9aa459adc8f6ba21f35ee41568d2686dfe71d16ccbcdc28546de77a",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-aarch64-glibc2_31-runtime-jetson.tar.gz"],
        sysroot_features = ["jetson"],
        disabled_for_features = [],
    )
    sysroot_repo(
        name = "sysroot_aarch64_glibc2_31_build_jetson",
        target_arch = "aarch64",
        variant = "build",
        libc_version = "glibc2_31",
        sha256 = "dfc378d0c18b4a48e045440064ecbe921d962c7e248fa206f9dc8b2d3754be44",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-aarch64-glibc2_31-build-jetson.tar.gz"],
        sysroot_features = ["jetson"],
        disabled_for_features = [],
    )
    sysroot_repo(
        name = "sysroot_aarch64_glibc2_31_test_jetson",
        target_arch = "aarch64",
        variant = "test",
        libc_version = "glibc2_31",
        sha256 = "d7b612b55408f59de36261b9a1b6f155840ed352bbd5262493ef9b0024886b65",
        urls = ["https://storage.googleapis.com/gimlet-dev-infra-public/sysroots/20231108122317/sysroot-aarch64-glibc2_31-test-jetson.tar.gz"],
        sysroot_features = ["jetson"],
        disabled_for_features = ["debug"],
    )

create_sysroots = _create_sysroots
