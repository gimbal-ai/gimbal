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

load("@bazel_gazelle//:def.bzl", "gazelle")

_GAZELLE_COMMON_ATTRS = dict(
    args = [
        "-proto_configs=$(location //bazel:rules_proto_config.yaml)",
        "-proto_imports_in=$(location @com_github_gogo_protobuf_proto//:imports.csv)",
    ],
    data = [
        "//bazel:rules_proto_config.yaml",
        "@com_github_gogo_protobuf_proto//:imports.csv",
    ],
    prefix = "gimletlabs.ai/gimlet",
)

def _gazelle_runners(**kwargs):
    attrs = dict(_GAZELLE_COMMON_ATTRS, **kwargs)
    gazelle(
        name = "gazelle",
        mode = "fix",
        **attrs
    )

    gazelle(
        name = "gazelle_diff",
        mode = "diff",
        **attrs
    )

gazelle_runners = _gazelle_runners
