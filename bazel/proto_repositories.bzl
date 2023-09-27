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

load("@build_stack_rules_proto//rules/proto:proto_repository.bzl", "proto_repository")
load("//bazel:lib.bzl", "default_arg")
load("//bazel:repository_locations.bzl", "PROTO_REPOSITORIES")

def _proto_repository_impl(name, **kwargs):
    default_arg(kwargs, "cfgs", ["//bazel:rules_proto_config.yaml"])
    default_arg(kwargs, "build_file_expunge", True)
    default_arg(kwargs, "build_file_proto_mode", "file")
    default_arg(kwargs, "build_directives", [
        "gazelle:go_generate_proto false",
    ])
    default_arg(kwargs, "languages", ["go", "proto", "protobuf"])

    proto_repository(
        name = name,
        **kwargs
    )

def _gml_proto_deps():
    for key, kwargs in PROTO_REPOSITORIES.items():
        _proto_repository_impl(key + "_proto", **kwargs)

gml_proto_deps = _gml_proto_deps
