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

load("@aspect_rules_js//js:defs.bzl", "js_run_binary", aspect_js_library = "js_library")
load("@aspect_rules_ts//ts:defs.bzl", aspect_ts_project = "ts_project")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")
load("//bazel:lib.bzl", "default_arg")

def _ts_project(name, **kwargs):
    default_arg(kwargs, "transpiler", "tsc")
    default_arg(kwargs, "declaration", True)
    default_arg(kwargs, "incremental", True)
    default_arg(kwargs, "allow_js", True)
    default_arg(kwargs, "preserve_jsx", True)
    default_arg(kwargs, "resolve_json_module", True)
    default_arg(kwargs, "tsconfig", "//src/ui:tsconfig")

    aspect_ts_project(name = name, **kwargs)

def _web_assets(name, **kwargs):
    aspect_js_library(name = name, **kwargs)

def _next_export(name, **kwargs):
    default_arg(kwargs, "out_dirs", ["out"])
    default_arg(kwargs, "tool", "//bazel/ui:next_js_binary")
    default_arg(kwargs, "args", ["build"])
    default_arg(kwargs, "copy_srcs_to_bin", True)
    default_arg(kwargs, "chdir", native.package_name())

    js_run_binary(name = name + "_next_build", **kwargs)

    pkg_tar(
        name = name,
        srcs = [
            ":" + name + "_next_build",
        ],
        strip_prefix = "out",
    )

def _next_standalone(name, **kwargs):
    default_arg(kwargs, "outs", ["out.tar"])
    default_arg(kwargs, "tool", "//bazel/ui:next_js_binary")
    default_arg(kwargs, "args", ["build"])
    default_arg(kwargs, "copy_srcs_to_bin", True)
    default_arg(kwargs, "chdir", native.package_name())

    js_run_binary(name = name, **kwargs)

NEXT_STANDALONE_CLEANUP = """
if [[ -d .next/standalone ]]; then
    find .next/standalone/node_modules/.aspect_rules_js -type l | while read line; do
        resolved="$(readlink -f "$line")"
        path_from_store="${resolved#*/.aspect_rules_js/}"
        rel="$(echo "$path_from_store" | sed 's|[^/]||g;s|/|../|g')"
        relative_symlink="${rel}${path_from_store}"
        ln -snf "$relative_symlink" "$line"
    done

    find .next/standalone/node_modules -type l -maxdepth 1 -not -path '*/.aspect_rules_js' | while read line; do
        resolved="$(readlink -f "$line")"
        relative_symlink=".aspect_rules_js/${resolved#*/.aspect_rules_js/}"
        ln -snf "$relative_symlink" "$line"
    done

    # pkg_tar doesn't support preserving symlinks so we create the tar here.
    mkdir out
    mkdir out/app
    mv .next/standalone/* out/app
    mv .next/standalone/.next out/app
    mv .next/static out/app/.next/static
    tar -C out -cf out.tar .
fi
"""

ts_project = _ts_project
web_assets = _web_assets
next_export = _next_export
next_standalone = _next_standalone
