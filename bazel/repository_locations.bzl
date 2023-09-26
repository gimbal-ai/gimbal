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

REPOSITORY_LOCATIONS = dict(
    aspect_rules_js = dict(
        sha256 = "77c4ea46c27f96e4aadcc580cd608369208422cf774988594ae8a01df6642c82",
        strip_prefix = "rules_js-1.32.2",
        urls = [
            "https://github.com/aspect-build/rules_js/releases/download/v1.32.2/rules_js-v1.32.2.tar.gz",
        ],
    ),
    aspect_rules_ts = dict(
        sha256 = "",
        strip_prefix = "rules_ts-2.0.0",
        urls = [
            "https://github.com/aspect-build/rules_ts/releases/download/v2.0.0/rules_ts-v2.0.0.tar.gz",
        ],
    ),
    aspect_bazel_lib = dict(
        sha256 = "09b51a9957adc56c905a2c980d6eb06f04beb1d85c665b467f659871403cf423",
        strip_prefix = "bazel-lib-1.34.5",
        urls = [
            "https://github.com/aspect-build/bazel-lib/releases/download/v1.34.5/bazel-lib-v1.34.5.tar.gz",
        ],
    ),
    aspect_rules_jest = dict(
        sha256 = "098186ffc450f2a604843d8ba14217088a0e259ea6a03294af5360a7f1bcd3e8",
        strip_prefix = "rules_jest-0.19.5",
        urls = [
            "https://github.com/aspect-build/rules_jest/releases/download/v0.19.5/rules_jest-v0.19.5.tar.gz",
        ],
    ),
    bazel_gazelle = dict(
        sha256 = "d3fa66a39028e97d76f9e2db8f1b0c11c099e8e01bf363a923074784e451f809",
        urls = [
            "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.33.0/bazel-gazelle-v0.33.0.tar.gz",
        ],
    ),
    rules_pkg = dict(
        sha256 = "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
        ],
    ),
    com_github_benchsci_rules_nodejs_gazelle = dict(
        sha256 = "738c139545e170daeeeb338879fac627648958f8ff07716d121de63b6599f3a6",
        strip_prefix = "rules_nodejs_gazelle-0.4.0",
        urls = [
            "https://github.com/benchsci/rules_nodejs_gazelle/archive/refs/tags/v0.4.0.tar.gz",
        ],
    ),
)
