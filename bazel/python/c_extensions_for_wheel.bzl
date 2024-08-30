# Copyright 2023- Gimlet Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@bazel_skylib//lib:sets.bzl", "sets")

def _c_extensions_for_wheel_impl(ctx):
    lib_path = "." + ctx.attr.pkg_name.replace("-", "_") + "_libs"

    inputs = []
    outputs = []

    input_libs = []
    output_paths = []
    runpaths = []

    # Handle each C extension lib.
    deps = sets.make()
    runfiles = []

    for ext, out_path in ctx.attr.extensions.items():
        ext_lib = ext[DefaultInfo].files.to_list()[0]
        inputs.append(ext_lib)
        input_libs.append(ext_lib.path)

        ext_output = ctx.actions.declare_file(out_path)
        outputs.append(ext_output)
        output_paths.append(ext_output.path)

        run_path = "/".join(["$ORIGIN"] + [".."] * out_path.count("/") + [lib_path])
        runpaths.append(run_path)

        runfiles.append(ext[DefaultInfo].default_runfiles.files)
        for file in ext[DefaultInfo].default_runfiles.files.to_list():
            if file == ext_lib:
                continue
            sets.insert(deps, file)

    # Handle all transitive dep libs.
    for file in sets.to_list(deps):
        input_libs.append(file)

        new_path = paths.join(lib_path, paths.basename(file.path))
        out = ctx.actions.declare_file(new_path)
        outputs.append(out)
        output_paths.append(out.path)

        runpaths.append("")

    args = ctx.actions.args()
    args.add_all("--input_libs", input_libs)
    args.add_all("--output_paths", output_paths)
    args.add_all("--runpaths", runpaths)

    ctx.actions.run(
        executable = ctx.attr._copy_shared_libs_tool[DefaultInfo].files_to_run,
        outputs = outputs,
        inputs = depset(
            inputs,
            transitive = runfiles,
        ),
        arguments = [args],
        mnemonic = "CExtForWheel",
    )

    return DefaultInfo(
        files = depset(
            outputs,
        ),
        # Keep the old runfiles so that the output shared lib still works within bazel.
        runfiles = ctx.runfiles(transitive_files = depset(transitive = runfiles)),
    )

c_extensions_for_wheel = rule(
    implementation = _c_extensions_for_wheel_impl,
    attrs = dict(
        extensions = attr.label_keyed_string_dict(
            mandatory = True,
            doc = "Mapping of C extensions to their output paths",
        ),
        pkg_name = attr.string(
            mandatory = True,
            doc = "Name of package the C extension will be in",
        ),
        _copy_shared_libs_tool = attr.label(
            default = Label("@gml//bazel/python:copy_shared_libs"),
            cfg = "exec",
        ),
    ),
)
