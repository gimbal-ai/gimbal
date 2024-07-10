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

load("@bazel_tools//tools/cpp:toolchain_utils.bzl", "find_cpp_toolchain", "use_cpp_toolchain")
load("@rules_cc//cc:action_names.bzl", "CPP_COMPILE_ACTION_NAME")

_EXTRA_COPTS = [
    "-x",
    "c++",
]

_HDR_EXTRA_COPTS = [
    "-Wno-pragma-once-outside-header",
    "-Wno-unused-const-variable",
    "-Wno-unused-function",
]

def _is_header(path):
    return path.endswith(".h") or path.endswith(".hpp") or path.endswith(".hh") or path.endswith(".hxx")

def _is_regular_c_src(path):
    return path.endswith(".cc") or path.endswith(".c") or path.endswith(".cpp")

def _generate_compile_commands(ctx, cc_toolchain, feature_configuration):
    """Generates compile commands for all the srcs in the given library."""
    clang_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = CPP_COMPILE_ACTION_NAME,
    )

    cc_info = ctx.attr.library[CcInfo]
    compilation_context = cc_info.compilation_context
    compile_variables = cc_common.create_compile_variables(
        feature_configuration = feature_configuration,
        cc_toolchain = cc_toolchain,
        user_compile_flags = ctx.attr.copts + ctx.fragments.cpp.copts,
        include_directories = compilation_context.includes,
        quote_include_directories = compilation_context.quote_includes,
        system_include_directories = compilation_context.system_includes,
        framework_include_directories = compilation_context.framework_includes,
        preprocessor_defines = depset(compilation_context.defines.to_list() + compilation_context.local_defines.to_list()),
    )

    # TODO(james): currently this will only work with c++ files, pure C files will be compiled as if they're c++ files.
    base_command_line = cc_common.get_memory_inefficient_command_line(
        feature_configuration = feature_configuration,
        action_name = CPP_COMPILE_ACTION_NAME,
        variables = compile_variables,
    )
    base_command_line = [clang_path] + base_command_line + _EXTRA_COPTS

    # Separate srcs from hdrs so we can add header only copts.
    hdrs = []
    srcs = []
    for src in ctx.attr.srcs:
        if len(src.files.to_list()) > 1:
            fail("clang_tidy rule currently only works with `srcs` with a single file per source target")
        path = src.files.to_list()[0].path
        if _is_header(path):
            hdrs.append(path)
        else:
            srcs.append(path)

    compile_commands_json = ctx.actions.declare_file(ctx.attr.name + ".compile_commands.json")

    gen_comp_script = ctx.actions.declare_file(ctx.attr.name + ".gen_comp_db.sh")
    ctx.actions.expand_template(
        template = ctx.attr._gen_comp_tpl.files.to_list()[0],
        output = gen_comp_script,
        substitutions = {
            "%hdr_cmdline%": str(base_command_line + _HDR_EXTRA_COPTS),
            "%hdrs%": " ".join(['"{}"'.format(p) for p in hdrs]),
            "%out_path%": compile_commands_json.path,
            "%src_cmdline%": str(base_command_line),
            "%srcs%": " ".join(['"{}"'.format(p) for p in srcs]),
            "%yq%": ctx.attr._yq.files.to_list()[0].path,
        },
        is_executable = True,
    )

    ctx.actions.run_shell(
        command = gen_comp_script.path,
        inputs = depset(ctx.attr._yq.files.to_list() + [gen_comp_script]),
        outputs = [compile_commands_json],
    )
    return compile_commands_json

def _generate_run_clang_tidy(ctx, feature_configuration, compile_commands_json):
    clang_tidy_path = cc_common.get_tool_for_action(
        feature_configuration = feature_configuration,
        action_name = "clang-tidy",
    )
    run_clang_tidy = ctx.actions.declare_file(ctx.attr.name + ".run_clang_tidy.sh")
    ctx.actions.expand_template(
        template = ctx.attr._run_ct_tpl.files.to_list()[0],
        output = run_clang_tidy,
        is_executable = True,
        substitutions = {
            "%clang_tidy%": clang_tidy_path,
            "%compile_commands%": compile_commands_json.path,
            "%yq%": ctx.attr._yq.files.to_list()[0].path,
        },
    )
    return run_clang_tidy

def _run_clang_tidy_on_files(ctx, cc_toolchain, run_clang_tidy, compile_commands_json):
    cc_info = ctx.attr.library[CcInfo]
    compilation_context = cc_info.compilation_context

    inputs = []
    for src in ctx.attr.srcs:
        inputs += src.files.to_list()
    inputs += compilation_context.headers.to_list()
    inputs.append(compile_commands_json)
    inputs += ctx.attr._yq.files.to_list()
    inputs += ctx.attr.clang_tidy_config.files.to_list()

    all_fixes = []
    for src in ctx.attr.srcs:
        f = src.files.to_list()[0]
        if not _is_header(f.path) and not _is_regular_c_src(f.path):
            continue

        fixes = ctx.actions.declare_file(ctx.attr.name + "." + f.basename + ".fixes.yaml")
        all_fixes.append(fixes)

        ctx.actions.run(
            executable = run_clang_tidy.path,
            arguments = [f.path, fixes.path],
            outputs = [fixes],
            mnemonic = "ClangTidy",
            inputs = depset(
                inputs + [run_clang_tidy],
                transitive = [cc_toolchain.all_files],
            ),
        )
    return all_fixes

def _clang_tidy_impl(ctx):
    cc_toolchain = find_cpp_toolchain(ctx)
    feature_configuration = cc_common.configure_features(
        ctx = ctx,
        cc_toolchain = cc_toolchain,
        requested_features = ctx.features,
        unsupported_features = ctx.disabled_features,
    )

    compile_commands_json = _generate_compile_commands(ctx, cc_toolchain, feature_configuration)
    run_clang_tidy = _generate_run_clang_tidy(ctx, feature_configuration, compile_commands_json)
    all_fixes = _run_clang_tidy_on_files(ctx, cc_toolchain, run_clang_tidy, compile_commands_json)

    return [
        DefaultInfo(
            files = depset(all_fixes + [compile_commands_json]),
        ),
        OutputGroupInfo(
            clang_tidy_fixes = depset(all_fixes),
            compile_commands = depset([compile_commands_json]),
        ),
    ]

clang_tidy = rule(
    implementation = _clang_tidy_impl,
    attrs = {
        "clang_tidy_config": attr.label(allow_single_file = True, default = Label("//:.clang-tidy")),
        "copts": attr.string_list(),
        "library": attr.label(providers = [CcInfo]),
        "srcs": attr.label_list(allow_files = True),
        "_cc_toolchain": attr.label(default = Label("@bazel_tools//tools/cpp:current_cc_toolchain")),
        "_gen_comp_tpl": attr.label(allow_single_file = True, default = Label("//bazel/clang_tidy:generate_compilation_database.sh.tpl")),
        "_run_ct_tpl": attr.label(allow_single_file = True, default = Label("//bazel/clang_tidy:run_clang_tidy.sh.tpl")),
        "_yq": attr.label(
            default = Label("@com_github_mikefarah_yq_v4//:v4"),
            cfg = "exec",
        ),
    },
    toolchains = use_cpp_toolchain(),
    fragments = ["cpp"],
)
