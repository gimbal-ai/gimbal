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
load("//bazel:deb_archive.bzl", "deb_archive")
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

def _deb_repo(name, **kwargs):
    existing_rule_keys = native.existing_rules().keys()
    if name in existing_rule_keys:
        # This repository has already been defined, probably because the user
        # wants to override the version. Do nothing.
        return

    location = REPOSITORY_LOCATIONS[name]

    deb_archive(
        name = name,
        urls = location["urls"],
        sha256 = location["sha256"],
        strip_prefix = location.get("strip_prefix", ""),
        **kwargs
    )

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
    # Pinned transitive dependencies.
    _bazel_repo("bazel_toolchains")
    _bazel_repo("upb")
    _bazel_repo(
        "cpuinfo",
        patches = ["//bazel/external:cpuinfo.fix_platforms.patch"],
        patch_args = ["-p1"],
    )
    _bazel_repo(
        "XNNPACK",
        patches = ["//bazel/external:xnnpack.fix_platforms.patch"],
        patch_args = ["-p1"],
    )

    # Dependencies with native bazel build files.
    _bazel_repo("com_google_protobuf", patches = ["//bazel/external:protobuf_gogo_hack.patch", "//bazel/external:protobuf_warning.patch"], patch_args = ["-p1"])
    _bazel_repo("com_github_grpc_grpc", patches = ["//bazel/external:grpc.patch", "//bazel/external:grpc_test_visibility.patch"], patch_args = ["-p1"])
    _bazel_repo("boringssl")

    _bazel_repo("com_github_gflags_gflags")
    _bazel_repo("com_google_flatbuffers")
    _bazel_repo(
        "org_tensorflow",
        repo_mapping = {
            "@python": "@python_3_9",
        },
    )
    _bazel_repo("com_github_neargye_magic_enum")
    _bazel_repo("com_github_thoughtspot_threadstacks")
    _bazel_repo("com_intel_tbb")
    _bazel_repo(
        "io_opentelemetry_cpp",
        patches = [
            # Ensure opentelemetry-cpp uses our vendored OTel protos instead of their own bazel definitions.
            "//bazel/external:opentelemetry_cpp.our_proto.patch",
        ],
        patch_args = ["-p1"],
    )

    # Dependencies where we provide an external BUILD file.
    _bazel_repo("com_github_arun11299_cpp_jwt", build_file = "//bazel/external:cpp_jwt.BUILD")
    _bazel_repo("com_github_nlohmann_json", build_file = "//bazel/external:nlohmann_json.BUILD")
    _bazel_repo("com_github_rlyeh_sole", patches = ["//bazel/external:sole.patch"], patch_args = ["-p1"], build_file = "//bazel/external:sole.BUILD")

    # Dependencies used in foreign cc rules (e.g. cmake-based builds)
    _include_all_repo("com_github_gperftools_gperftools")
    _include_all_repo("com_github_nats_io_natsc")
    _bazel_repo(
        "build_stack_rules_proto",
        patches = [
            "//bazel/external:rules_proto.silence_warnings.patch",
            "//bazel/external:rules_proto.cpp_plugin_fix.patch",
        ],
        patch_args = ["-p1"],
    )
    _include_all_repo("com_github_libuv_libuv", patches = ["//bazel/external:libuv.patch"], patch_args = ["-p1"])

    # NVIDIA deps.
    _bazel_repo("nvidia_stubs", build_file = "//bazel/external:nvidia_stubs.BUILD")
    _bazel_repo("com_gitlab_nvidia_headers_cudart", build_file = "//bazel/external:cudart.BUILD")
    _bazel_repo("com_gitlab_nvidia_headers_nvcc", build_file = "//bazel/external:nvcc.BUILD")
    _bazel_repo("com_github_nvidia_tensorrt", build_file = "//bazel/external:tensorrt.BUILD")
    _bazel_repo("com_github_onnx_onnx_tensorrt", build_file = "//bazel/external:onnx_tensorrt.BUILD")
    _deb_repo("com_nvidia_jetson_multimedia_api", build_file = "//bazel/external:jetson_multimedia_api.BUILD")
    _deb_repo("com_nvidia_jetson_multimedia_utils", build_file = "//bazel/external:jetson_multimedia_utils.BUILD")
    _deb_repo("com_nvidia_l4t_camera", build_file = "//bazel/external:l4t_camera.BUILD")

    # mediapipe deps.
    _bazel_repo(
        "com_github_ffmpeg_ffmpeg",
        patches = [
            "//bazel/external:ffmpeg.fix_configure.patch",
        ],
        patch_args = ["-p1"],
        build_file = "//bazel/external:ffmpeg.BUILD",
    )
    _bazel_repo("com_github_opencv_opencv", build_file = "//bazel/external:opencv.BUILD")
    _bazel_repo(
        "com_github_google_mediapipe",
        patches = [
            # Use the opencv we build.
            "//bazel/external:mediapipe.our_opencv.patch",
            # Only generate cc and/or go protos.
            "//bazel/external:mediapipe.disable_extra_protos.patch",
            # Use the ffmpeg we build.
            "//bazel/external:mediapipe.our_ffmpeg.patch",
            # Make mediapipe compatible with our version of opencv.
            "//bazel/external:mediapipe.opencv4_fix.patch",
            # Make the mediapipe hand tracking example visible for testing purposes.
            # TODO(james): remove this once we have our own usage of mediapipe.
            "//bazel/external:mediapipe.example_visibility.patch",
        ],
        patch_args = ["-p1"],
        repo_mapping = {
            "@com_github_glog_glog": "@com_github_google_glog",
            "@mediapipe": "@com_github_google_mediapipe",
            "@npm": "@mediapipe_npm",
        },
    )

    _bazel_repo(
        "com_github_cisco_openh264",
        build_file = "//bazel/external:openh264.BUILD",
        patches = ["//bazel/external:openh264.version_gen.patch"],
        patch_args = ["-p1"],
    )

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
    cc_toolchain_config_repo("unix_cc_toolchain_config", patch = "//bazel/cc_toolchains:unix_cc_toolchain_config.patch")

def _gml_deps():
    _bazel_repo("bazel_skylib")
    _bazel_repo("com_github_fmeum_rules_meta")

    _bazel_repo(
        "com_github_benchsci_rules_nodejs_gazelle",
        patches = [
            "//bazel/external:rules_nodejs_gazelle.builtins.patch",
            "//bazel/external:rules_nodejs_gazelle.import.patch",
            "//bazel/external:rules_nodejs_gazelle.snapshots.patch",
        ],
        patch_args = ["-p1"],
    )

    _com_llvm_lib()
    _cc_deps()

list_gml_deps = _list_gml_deps
gml_deps = _gml_deps
gml_cc_toolchain_deps = _gml_cc_toolchain_deps
