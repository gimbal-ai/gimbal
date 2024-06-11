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

load("@rules_python//python:defs.bzl", "PyInfo")

def _cleanup_import(sys_path_prefix):
    if sys_path_prefix.startswith("_main/"):
        prefix = sys_path_prefix.removeprefix("_main/")
    else:
        prefix = "../" + sys_path_prefix
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix

def _remove_import_prefix(imports, path):
    for import_prefix in imports:
        if path.startswith(import_prefix):
            return path.removeprefix(import_prefix)
    return path

_script_tpl = """
#!/bin/bash -e

copy_and_fix_proto() {{
    dir="$(dirname "$2")"
    mkdir -p "${{dir}}"
    touch "${{dir}}/__init__.py"
    awk 'matched=0; \
        /^from google.protobuf./{{ matched=1; print }} \
        (!matched) && match($0, /^from .*/) \
            {{ matched=1; print "from {import_prefix}." substr($0, RSTART+5, RLENGTH-5) }} \
        (!matched) {{ print }}' "$1" > "$2"
}}
"""

def _copy_python_protos_impl(ctx):
    src_dest = []

    transitive_imports = []
    transitive_files = []
    transitive_runfiles = []

    for pkg in ctx.attr.srcs:
        imports = [
            _cleanup_import(sys_path_prefix)
            for sys_path_prefix in pkg[PyInfo].imports.to_list()
        ]
        transitive_imports.append(pkg[PyInfo].imports)

        transitive_runfiles.append(pkg[DefaultInfo].default_runfiles)

        for src in pkg[PyInfo].transitive_sources.to_list():
            if not src.basename.endswith("pb2.py") and not src.basename.endswith("pb2_grpc.py"):
                transitive_files.append(src)
                continue
            path = _remove_import_prefix(imports, src.short_path)
            if src.short_path.startswith("../"):
                transitive_files.append(src)
                continue
            dest = ctx.actions.declare_file("/".join([ctx.attr.out_dir, path]))
            src_dest.append((src, dest))

    sh_file = ctx.actions.declare_file(ctx.label.name + "_correct_imports.sh")
    ctx.actions.write(
        output = sh_file,
        content = _script_tpl.format(
            import_prefix = ctx.attr.import_prefix,
        ) + "\n".join([
            "copy_and_fix_proto {} {}".format(
                src.path,
                dest.path,
            )
            for src, dest in src_dest
        ]),
    )
    outputs = [dest for _, dest in src_dest]
    ctx.actions.run(
        outputs = outputs,
        inputs = [src for src, _ in src_dest],
        executable = sh_file,
    )
    transitive_depset = depset(transitive_files)
    runfiles = ctx.runfiles(files = outputs, transitive_files = transitive_depset)
    runfiles = runfiles.merge_all(transitive_runfiles)
    return [
        DefaultInfo(
            files = depset(outputs),
            runfiles = runfiles,
        ),
        PyInfo(
            transitive_sources = depset(outputs, transitive = [transitive_depset]),
            imports = depset(["."], transitive = transitive_imports),
        ),
    ]

copy_python_protos = rule(
    doc = """
    Recursively copies all proto generated python files created by the `srcs` to the package the rule is in.
    """,
    attrs = dict(
        srcs = attr.label_list(
            mandatory = True,
            providers = [PyInfo],
            allow_files = True,
            cfg = "exec",
        ),
        import_prefix = attr.string(mandatory = True),
        out_dir = attr.string(mandatory = True),
    ),
    implementation = _copy_python_protos_impl,
)
