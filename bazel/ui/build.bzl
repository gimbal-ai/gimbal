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

load("@aspect_rules_jest//jest:defs.bzl", aspect_jest_test = "jest_test")
load("@aspect_rules_js//js:defs.bzl", "js_run_binary", aspect_js_library = "js_library")
load("@aspect_rules_ts//ts:defs.bzl", aspect_ts_project = "ts_project")
load("//bazel:gml_build_system.bzl", "no_sysroot")
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

def _jest_test(name = "", srcs = [], deps = [], data = [], **kwargs):
    node_modules = kwargs.pop("node_modules", "//src/ui:node_modules")
    tags = kwargs.pop("tags", ["jest"])

    ts_project(
        name = "%s_js" % name,
        srcs = srcs,
        deps = deps,
    )

    seen = {
        # The nextjs jest setup complains if neither the app nor the pages dir exist.
        # Copying over the layout file from app ensures that the app dir exists.
        "//src/ui/app:cp_layout": True,
        "//src/ui:cp_setup_jest": True,
        "//src/ui:cp_tsconfig": True,
    }
    for d in deps:
        seen[d] = True
    for d in data:
        seen[d] = True

    aspect_jest_test(
        name = name,
        data = srcs + seen.keys(),
        node_modules = node_modules,
        tags = tags,
        target_compatible_with = no_sysroot(),
        **kwargs
    )

def _web_assets(name, **kwargs):
    aspect_js_library(name = name, **kwargs)

NEXTJS_STANDALONE_SCRIPT = """
set -e;
export PATH="$$PATH:/opt/gml_dev/tools/node/bin";
pushd src/ui &> /dev/null;
pnpm install --frozen-lockfile &> /dev/null;
pnpm build &> /dev/null;
out="$$(mktemp -d)";
mkdir -p "$${out}/app"
cp -r .next/standalone/* "$${out}/app";
cp -r .next/standalone/.next "$${out}/app";
cp -r .next/static "$${out}/app/.next/static";
popd &> /dev/null;
tar --mtime="2023-01-01 00:00:00 UTC" -C "$${out}" -cf "$(location standalone.tar)" .;
rm -rf "$${out}";
"""

def _next_standalone(name, **kwargs):
    default_arg(kwargs, "srcs", [])
    default_arg(kwargs, "outs", ["standalone.tar"])

    # We use a genrule with sandboxing disabled since the nextjs tracing for standalone builds
    # broke in next v13.5.4
    # Once this is fixed, we can go back to building in the sandbox.
    # See https://github.com/vercel/next.js/issues/62591
    native.genrule(
        name = name,
        cmd = NEXTJS_STANDALONE_SCRIPT,
        local = True,
        **kwargs
    )

def _storybook_export(name, **kwargs):
    default_arg(kwargs, "outs", ["storybook.tar"])
    default_arg(kwargs, "tool", "//bazel/ui:storybook_binary")
    default_arg(kwargs, "args", ["build", "--disable-telemetry"])
    default_arg(kwargs, "copy_srcs_to_bin", True)
    default_arg(kwargs, "chdir", native.package_name())
    default_arg(kwargs, "env", dict())

    kwargs["env"]["PATH"] = "/bin:/usr/bin:/usr/local/bin:/opt/gml_dev/tools/node/bin"
    kwargs["srcs"] = kwargs["srcs"]

    js_run_binary(name = name, **kwargs)

# This should be possible with just using package_tar but something about the storybook
# output makes bazel not like the output dir. Using our trusted postrun works around this
# issue
STORYBOOK_CLEANUP = """
if [[ -d storybook-static ]]; then
    tar --mtime="2023-01-01 00:00:00 UTC" -C storybook-static -cf storybook.tar .
fi
"""

ts_project = _ts_project
web_assets = _web_assets
next_standalone = _next_standalone
jest_test = _jest_test
storybook_export = _storybook_export
