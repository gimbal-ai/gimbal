# Copyright 2018- The Pixie Authors.
# Modifications Copyright 2023- Gimlet Labs, Inc.
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

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("//bazel/cc_toolchains:deps.bzl", "cc_toolchain_config_repo")
load(":repository_locations.bzl", "REPOSITORY_LOCATIONS")

# Make all contents of an external repository accessible under a filegroup.
# Used for external HTTP archives, e.g. cares.
BUILD_ALL_CONTENT = """filegroup(name = "all", srcs = glob(["**"]), visibility = ["//visibility:public"])"""

def _http_archive_repo_impl(name, **kwargs):
    # `existing_rule_keys` contains the names of repositories that have already
    # been defined in the Bazel workspace. By skipping repos with existing keys,
    # users can override dependency versions by using standard Bazel repository
    # rules in their WORKSPACE files.
    existing_rule_keys = native.existing_rules().keys()
    if name in existing_rule_keys:
        # This repository has already been defined, probably because the user
        # wants to override the version. Do nothing.
        return

    location = REPOSITORY_LOCATIONS[name]

    # HTTP tarball at a given URL. Add a BUILD file if requested.
    http_archive(
        name = name,
        urls = location["urls"],
        sha256 = location["sha256"],
        strip_prefix = location.get("strip_prefix", ""),
        **kwargs
    )

# For bazel repos do not require customization.
def _bazel_repo(name, **kwargs):
    _http_archive_repo_impl(name, **kwargs)

# With a predefined "include all files" BUILD file for a non-Bazel repo.
def _include_all_repo(name, **kwargs):
    kwargs["build_file_content"] = BUILD_ALL_CONTENT
    _http_archive_repo_impl(name, **kwargs)

def _com_llvm_lib():
    _bazel_repo("com_llvm_lib_x86_64_glibc_host", build_file = "//bazel/external:llvm.BUILD")
    _bazel_repo("com_llvm_lib_libcpp_x86_64_glibc_host", build_file = "//bazel/external:llvm.BUILD")
    _bazel_repo("com_llvm_lib_libcpp_x86_64_glibc_host_asan", build_file = "//bazel/external:llvm.BUILD")
    _bazel_repo("com_llvm_lib_libcpp_x86_64_glibc_host_msan", build_file = "//bazel/external:llvm.BUILD")
    _bazel_repo("com_llvm_lib_libcpp_x86_64_glibc_host_tsan", build_file = "//bazel/external:llvm.BUILD")

    _bazel_repo("com_llvm_lib_x86_64_glibc2_36", build_file = "//bazel/external:llvm.BUILD")
    _bazel_repo("com_llvm_lib_libcpp_x86_64_glibc2_36", build_file = "//bazel/external:llvm.BUILD")

    _bazel_repo("com_llvm_lib_aarch64_glibc2_36", build_file = "//bazel/external:llvm.BUILD")
    _bazel_repo("com_llvm_lib_libcpp_aarch64_glibc2_36", build_file = "//bazel/external:llvm.BUILD")

def _cc_deps():
    # Dependencies with native bazel build files.

    _bazel_repo("upb")
    _bazel_repo("com_google_protobuf", patches = ["//bazel/external:protobuf_gogo_hack.patch", "//bazel/external:protobuf_text_format.patch", "//bazel/external:protobuf_warning.patch"], patch_args = ["-p1"])
    _bazel_repo("com_github_grpc_grpc", patches = ["//bazel/external:grpc.patch", "//bazel/external:grpc_go_toolchain.patch", "//bazel/external:grpc_test_visibility.patch"], patch_args = ["-p1"])
    _bazel_repo("boringssl")

    _bazel_repo("com_google_benchmark")
    _bazel_repo("com_google_googletest")
    _bazel_repo("com_github_gflags_gflags")
    _bazel_repo("com_github_google_glog")
    _bazel_repo("com_google_absl")
    _bazel_repo("com_google_flatbuffers")
    _bazel_repo("org_tensorflow", patches = ["//bazel/external:tensorflow_disable_llvm.patch", "//bazel/external:tensorflow_disable_mirrors.patch", "//bazel/external:tensorflow_disable_py.patch"], patch_args = ["-p1"])
    _bazel_repo("com_github_neargye_magic_enum")
    _bazel_repo("com_github_thoughtspot_threadstacks")
    _bazel_repo("com_googlesource_code_re2", patches = ["//bazel/external:re2_warning.patch"], patch_args = ["-p1"])
    _bazel_repo("com_intel_tbb")

    # Dependencies where we provide an external BUILD file.
    _bazel_repo("com_github_arun11299_cpp_jwt", build_file = "//bazel/external:cpp_jwt.BUILD")
    _bazel_repo("com_github_nlohmann_json", build_file = "//bazel/external:nlohmann_json.BUILD")
    _bazel_repo("com_github_rlyeh_sole", patches = ["//bazel/external:sole.patch"], patch_args = ["-p1"], build_file = "//bazel/external:sole.BUILD")

    # Dependencies used in foreign cc rules (e.g. cmake-based builds)
    _include_all_repo("com_github_gperftools_gperftools")
    _include_all_repo("com_github_nats_io_natsc")

def _list_gml_deps(name):
    modules = dict()
    for _, repo_config in REPOSITORY_LOCATIONS.items():
        if "manual_license_name" in repo_config:
            modules["#manual-license-name:" + repo_config["manual_license_name"]] = True
            continue
        urls = repo_config["urls"]
        best_url = None
        for url in urls:
            if url.startswith("https://github.com") or best_url == None:
                best_url = url
        modules[best_url] = True

    module_lines = []
    for key in modules.keys():
        module_lines.append(key)

    native.genrule(
        name = name,
        outs = ["{}.out".format(name)],
        cmd = 'echo "{}" > $@'.format("\n".join(module_lines)),
        visibility = ["//visibility:public"],
    )

def _gml_cc_toolchain_deps():
    _bazel_repo("bazel_skylib")
    cc_toolchain_config_repo("unix_cc_toolchain_config", patch = "//bazel/cc_toolchains:unix_cc_toolchain_config.patch")

def _gml_deps():
    _bazel_repo("rules_pkg")
    _bazel_repo("bazel_gazelle")

    _bazel_repo("io_bazel_rules_go")
    _bazel_repo("rules_foreign_cc")
    _bazel_repo("rules_python")
    _bazel_repo("com_github_bazelbuild_buildtools")
    _bazel_repo("com_github_fmeum_rules_meta")

    _bazel_repo("com_github_benchsci_rules_nodejs_gazelle", patches = ["//bazel/external:rules_nodejs_gazelle.patch"], patch_args = ["-p1"])
    _bazel_repo("com_github_benchsci_rules_nodejs_gazelle")
    _bazel_repo("aspect_bazel_lib")
    _bazel_repo("aspect_rules_js", patches = ["//bazel/external:rules_js.patch"], patch_args = ["-p1"])
    _bazel_repo("aspect_rules_ts")
    _bazel_repo("aspect_rules_jest")

    _com_llvm_lib()
    _cc_deps()

list_gml_deps = _list_gml_deps
gml_deps = _gml_deps
gml_cc_toolchain_deps = _gml_cc_toolchain_deps
