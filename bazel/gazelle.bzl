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
load("@rules_python_gazelle_plugin//:def.bzl", "GAZELLE_PYTHON_RUNTIME_DEPS")

_GAZELLE_COMMON_ATTRS = dict(
    args = [
        "-proto_configs=$(location //bazel:rules_proto_config.yaml)",
    ],
    data = GAZELLE_PYTHON_RUNTIME_DEPS + [
        "//bazel:rules_proto_config.yaml",
    ],
    prefix = "gimletlabs.ai/gimlet",
)

def _gazelle_runners(**kwargs):
    attrs = dict(_GAZELLE_COMMON_ATTRS, **kwargs)
    gazelle(
        name = "gazelle",
        command = "fix",
        mode = "fix",
        **attrs
    )

    gazelle(
        name = "gazelle_diff",
        command = "fix",
        mode = "diff",
        **attrs
    )

gazelle_runners = _gazelle_runners
