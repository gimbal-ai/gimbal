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

REPOSITORY_LOCATIONS = dict(
    # Must be called boringssl to make sure the deps pick it up correctly.
    boringssl = dict(
        sha256 = "d11f382c25a3bea34ad8761d57828971c8b06e230ad99e1cbfd4253c419f4f9a",
        strip_prefix = "boringssl-7b00d84b025dff0c392c2df5ee8aa6d3c63ad539",
        urls = ["https://github.com/google/boringssl/" +
                "archive/7b00d84b025dff0c392c2df5ee8aa6d3c63ad539.tar.gz"],
    ),
    com_github_arun11299_cpp_jwt = dict(
        sha256 = "6dbf93969ec48d97ecb6c157014985846df8c01995a0011c21f4e2c146594922",
        strip_prefix = "cpp-jwt-1.1.1",
        urls = ["https://github.com/arun11299/cpp-jwt/archive/refs/tags/v1.1.1.tar.gz"],
    ),
    com_github_benchsci_rules_nodejs_gazelle = dict(
        sha256 = "a96a4be31fbf53669a866c66be90ce686a73ee323926443c7c9a54bb100e4ff1",
        strip_prefix = "rules_nodejs_gazelle-0.4.1",
        urls = [
            "https://github.com/benchsci/rules_nodejs_gazelle/archive/refs/tags/v0.4.1.tar.gz",
        ],
    ),
    com_github_fmeum_rules_meta = dict(
        sha256 = "ed3ed909e6e3f34a11d7c2adcc461535975a875fe434719540a4e6f63434a866",
        strip_prefix = "rules_meta-0.0.4",
        urls = [
            "https://github.com/fmeum/rules_meta/archive/refs/tags/v0.0.4.tar.gz",
        ],
    ),
    com_github_gflags_gflags = dict(
        sha256 = "9e1a38e2dcbb20bb10891b5a171de2e5da70e0a50fff34dd4b0c2c6d75043909",
        strip_prefix = "gflags-524b83d0264cb9f1b2d134c564ef1aa23f207a41",
        urls = ["https://github.com/gflags/gflags/archive/524b83d0264cb9f1b2d134c564ef1aa23f207a41.tar.gz"],
    ),
    com_github_gperftools_gperftools = dict(
        sha256 = "ea566e528605befb830671e359118c2da718f721c27225cbbc93858c7520fee3",
        strip_prefix = "gperftools-2.9.1",
        urls = ["https://github.com/gperftools/gperftools/releases/download/gperftools-2.9.1/gperftools-2.9.1.tar.gz"],
    ),
    com_github_grpc_grpc = dict(
        sha256 = "b55696fb249669744de3e71acc54a9382bea0dce7cd5ba379b356b12b82d4229",
        strip_prefix = "grpc-1.51.1",
        urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.51.1.tar.gz"],
    ),
    com_github_libuv_libuv = dict(
        sha256 = "371e5419708f6aaeb8656671f89400b92a9bba6443369af1bb70bcd6e4b3c764",
        strip_prefix = "libuv-1.42.0",
        urls = ["https://github.com/libuv/libuv/archive/refs/tags/v1.42.0.tar.gz"],
    ),
    com_github_nats_io_natsc = dict(
        sha256 = "16e700d912034faefb235a955bd920cfe4d449a260d0371b9694d722eb617ae1",
        strip_prefix = "nats.c-3.3.0",
        urls = ["https://github.com/nats-io/nats.c/archive/refs/tags/v3.3.0.tar.gz"],
    ),
    com_github_neargye_magic_enum = dict(
        sha256 = "4fe6627407a656d0d73879c0346b251ccdcfb718c37bef5410ba172c7c7d5f9a",
        strip_prefix = "magic_enum-0.7.0",
        urls = ["https://github.com/Neargye/magic_enum/archive/refs/tags/v0.7.0.tar.gz"],
    ),
    com_github_nlohmann_json = dict(
        sha256 = "87b5884741427220d3a33df1363ae0e8b898099fbc59f1c451113f6732891014",
        urls = ["https://github.com/nlohmann/json/releases/download/v3.7.3/include.zip"],
    ),
    com_github_rlyeh_sole = dict(
        sha256 = "70dbd71f2601963684195f4c7d8a1c2d45a0d53114bc4d06f8cebe6d3d3ffa69",
        strip_prefix = "sole-95612e5cda1accc0369a51edfe0f32bfb4bee2a0",
        urls = ["https://github.com/r-lyeh-archived/sole/archive/95612e5cda1accc0369a51edfe0f32bfb4bee2a0.tar.gz"],
    ),
    com_github_thoughtspot_threadstacks = dict(
        sha256 = "e54d4c3cd5af3cc136cc952c1ef77cd90b41133cd61140d8488e14c6d6f795e9",
        strip_prefix = "threadstacks-94adbe26c4aaf9ca945fd7936670d40ec6f228fb",
        urls = ["https://github.com/gimletlabs/threadstacks/archive/94adbe26c4aaf9ca945fd7936670d40ec6f228fb.tar.gz"],
    ),
    com_google_flatbuffers = dict(
        sha256 = "e2dc24985a85b278dd06313481a9ca051d048f9474e0f199e372fea3ea4248c9",
        strip_prefix = "flatbuffers-2.0.6",
        urls = ["https://github.com/google/flatbuffers/archive/refs/tags/v2.0.6.tar.gz"],
    ),
    com_google_protobuf = dict(
        sha256 = "63c5539a8506dc6bccd352a857cea106e0a389ce047a3ff0a78fe3f8fede410d",
        strip_prefix = "protobuf-24487dd1045c7f3d64a21f38a3f0c06cc4cf2edb",
        urls = [
            "https://storage.googleapis.com/grpc-bazel-mirror/github.com/protocolbuffers/protobuf/archive/24487dd1045c7f3d64a21f38a3f0c06cc4cf2edb.tar.gz",
            "https://github.com/protocolbuffers/protobuf/archive/24487dd1045c7f3d64a21f38a3f0c06cc4cf2edb.tar.gz",
        ],
    ),
    com_google_protobuf_javascript = dict(
        sha256 = "35bca1729532b0a77280bf28ab5937438e3dcccd6b31a282d9ae84c896b6f6e3",
        strip_prefix = "protobuf-javascript-3.21.2",
        urls = [
            "https://github.com/protocolbuffers/protobuf-javascript/archive/refs/tags/v3.21.2.tar.gz",
        ],
    ),
    com_intel_tbb = dict(
        sha256 = "91eab849ab1442db72317f8c968c5a1010f8546ca35f26086201262096c8a8a9",
        strip_prefix = "oneTBB-e6104c9599f7f10473caf545199f7468c0a8e52f",
        urls = ["https://github.com/oneapi-src/oneTBB/archive/e6104c9599f7f10473caf545199f7468c0a8e52f.tar.gz"],
    ),
    com_llvm_clang_15 = dict(
        sha256 = "3383bc528091462ec707ce21ef6d595d396ff6f04e6776b5735e5e1234d0a33d",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/clang-min-15.0-20231108142731.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_x86_64_glibc_host = dict(
        sha256 = "a242a754a4b26cf90872bf2d5bbc4d8518b60ebbcf8294ddef3a79ff362bd47a",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libstdc%2B%2B.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host = dict(
        sha256 = "bf5a068624251e6eeb7cddaaa7b04b2781a947bc3086d6f193d7ddffafa9641a",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libcxx.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host_asan = dict(
        sha256 = "89ddad42e60c63f4e81770a175325fa43d2ec9624d6502a11abaec1ef76946a9",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libcxx-asan.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host_msan = dict(
        sha256 = "4fc31a499eed4fa0acbe164fbb14c5ec7de5e56c325e565155a4811f4fe44449",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libcxx-msan.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host_tsan = dict(
        sha256 = "db11bbad23a4430ad660777c2bf5a0049f0243f3c021011c301d8c555e4939b7",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libcxx-tsan.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_x86_64_glibc2_36 = dict(
        sha256 = "191ae1ca61f09345f9cee71eef8bb11b2c5f5d4426095773374757ecb932153a",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libstdc++-x86_64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc2_36 = dict(
        sha256 = "fda709b8473d7cee09338b9fb316f313d0621b944350a213251eb5ff5e65e627",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libcxx-x86_64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_aarch64_glibc2_31 = dict(
        sha256 = "7c03edc88613e4f69d0b09cc5f7a4d466d83b4aa584079d226d07751a0561aca",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libcxx-jetson-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_aarch64_glibc2_36 = dict(
        sha256 = "48a28c1ba2ee1e93dcef38e6d19c4e74059bc046440e897bd00f6fea209c80e9",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libstdc++-aarch64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_aarch64_glibc2_36 = dict(
        sha256 = "5f0708bc3ce9c794ec879c4439909c4a8514236cf6c3ebfb0b44283eca8f4448",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/llvm-15.0-20231108142731-libcxx-aarch64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_x86_64_glibc_host = dict(
        sha256 = "ef0c8958d7fb3752d97a88496a70c591043360100c2aaccf120c0ce22a2ee3db",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/libcxx-15.0-20231108142731.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_x86_64_glibc2_36 = dict(
        sha256 = "c4119805a87b596848d2f6cfe90e5b873ec223962b3a07fe25e01b5640a01b3a",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/libcxx-15.0-20231108142731-x86_64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_aarch64_glibc2_36 = dict(
        sha256 = "c2254beb98c99d769542ad407654e7e5b30d2661c80af6e123b92b4b0395d275",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/libcxx-15.0-20231108142731-aarch64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_aarch64_glibc2_31 = dict(
        sha256 = "6a3c1b6d4094a42ff333ff3a6dea8d4bd44ef5e9a3896dada8673a76d70496ca",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20231108142731/libcxx-15.0-20231108142731-jetson-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    org_tensorflow = dict(
        sha256 = "0a7c004769fe726396a27e96ce0e494b297662305c7b4f2c749ea4c0ad5bc39a",
        strip_prefix = "tensorflow-3ee6de0b9ae296d0a5fc59815bdb776b3200263f",
        urls = ["https://github.com/gimletlabs/tensorflow/archive/3ee6de0b9ae296d0a5fc59815bdb776b3200263f.tar.gz"],
    ),
    unix_cc_toolchain_config = dict(
        sha256 = "2c1d60ef4d586909f138c28409780e102e2ebd619e7d462ded26dce43a8f9ffb",
        urls = [
            "https://raw.githubusercontent.com/bazelbuild/bazel/5.3.1/tools/cpp/unix_cc_toolchain_config.bzl",
        ],
    ),
    # GRPC and Protobuf pick different versions. Pick the newer one.
    upb = dict(
        sha256 = "017a7e8e4e842d01dba5dc8aa316323eee080cd1b75986a7d1f94d87220e6502",
        strip_prefix = "upb-e4635f223e7d36dfbea3b722a4ca4807a7e882e2",
        urls = [
            "https://storage.googleapis.com/grpc-bazel-mirror/github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
            "https://github.com/protocolbuffers/upb/archive/e4635f223e7d36dfbea3b722a4ca4807a7e882e2.tar.gz",
        ],
    ),
    build_stack_rules_proto = dict(
        sha256 = "ac7e2966a78660e83e1ba84a06db6eda9a7659a841b6a7fd93028cd8757afbfb",
        strip_prefix = "rules_proto-2.0.1",
        urls = [
            "https://github.com/stackb/rules_proto/archive/v2.0.1.tar.gz",
        ],
    ),
    nvidia_stubs = dict(
        sha256 = "eed79efc454c2493f3e1a6277be6d351ec33b4ba4c27306168898dd1d0480f46",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/nvidia-stubs/20231009012327/nvidia_stubs.tar.gz",
        ],
    ),
    com_gitlab_nvidia_headers_cudart = dict(
        sha256 = "0d1e2249f4e75c96a51d327a4cfcf95413e90b59855c09c738f5ed4a50df469c",
        strip_prefix = "cudart-cuda-11.4.4",
        urls = ["https://gitlab.com/nvidia/headers/cuda-individual/cudart/-/archive/cuda-11.4.4/cudart-cuda-11.4.4.tar.gz"],
    ),
    com_gitlab_nvidia_headers_nvcc = dict(
        sha256 = "677ec9463140e74c66c8936f18a8c658f6a3a440d3ccf54f9e4f3622a1853326",
        strip_prefix = "nvcc-cuda-11.4.4",
        urls = ["https://gitlab.com/nvidia/headers/cuda-individual/nvcc/-/archive/cuda-11.4.4/nvcc-cuda-11.4.4.tar.gz"],
    ),
    com_github_nvidia_tensorrt = dict(
        sha256 = "72a297e39d143dbe94592244903aab82ef4cd7c773ce7cf6b562d9355da9cf67",
        strip_prefix = "TensorRT-8.5.2",
        urls = ["https://github.com/NVIDIA/TensorRT/archive/refs/tags/8.5.2.tar.gz"],
    ),
    com_github_onnx_onnx_tensorrt = dict(
        sha256 = "5c90c8b65828af0079a8bc2189d0c6c161f1fc0b2522f2ac8b65aa30da42ccb2",
        strip_prefix = "onnx-tensorrt-release-8.5-GA",
        urls = ["https://github.com/onnx/onnx-tensorrt/archive/refs/tags/release/8.5-GA.tar.gz"],
    ),
    com_github_ffmpeg_ffmpeg = dict(
        sha256 = "5f417a4c00ec7874d255b24dbca33246be7ea72253bd9449bf9ebac51133e2a8",
        strip_prefix = "FFmpeg-n4.3.6",
        urls = [
            "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n4.3.6.tar.gz",
        ],
    ),
    com_github_opencv_opencv = dict(
        sha256 = "62f650467a60a38794d681ae7e66e3e8cfba38f445e0bf87867e2f2cdc8be9d5",
        strip_prefix = "opencv-4.8.1",
        urls = ["https://github.com/opencv/opencv/archive/refs/tags/4.8.1.tar.gz"],
    ),
    cpuinfo = dict(
        sha256 = "109e9d2f95a0d72fe50174b44de5b8c9be3e9551407b882e4ad497e9e068d202",
        strip_prefix = "cpuinfo-87d8234510367db49a65535021af5e1838a65ac2",
        urls = ["https://github.com/pytorch/cpuinfo/archive/87d8234510367db49a65535021af5e1838a65ac2.tar.gz"],
    ),
    XNNPACK = dict(
        sha256 = "104d3ef9efa30e97bf036194b43b5d4404d8522b960634424aa98ac4116b5a7d",
        strip_prefix = "XNNPACK-b9d4073a6913891ce9cbd8965c8d506075d2a45a",
        urls = ["https://github.com/google/XNNPACK/archive/b9d4073a6913891ce9cbd8965c8d506075d2a45a.tar.gz"],
    ),
    # Pull in a newer version of bazel toolchains for tensorflow.
    bazel_toolchains = dict(
        sha256 = "02e4f3744f1ce3f6e711e261fd322916ddd18cccd38026352f7a4c0351dbda19",
        strip_prefix = "bazel-toolchains-5.1.2",
        urls = ["https://github.com/bazelbuild/bazel-toolchains/archive/refs/tags/v5.1.2.tar.gz"],
    ),
    # if bazel_skylib is only pulled in as a bzlmod dep somewhere in the tensorflow dependency tree
    # a third party project will pull a very old version of bazel skylib that doesn't work.
    bazel_skylib = dict(
        sha256 = "66ffd9315665bfaafc96b52278f57c7e2dd09f5ede279ea6d39b2be471e7e3aa",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.2/bazel-skylib-1.4.2.tar.gz",
        ],
    ),
    com_github_google_mediapipe = dict(
        sha256 = "0746834dc40600ee75fcc6b18181b612094c25badafeeba7c53ca2f3f0813d46",
        strip_prefix = "mediapipe-ce4f6d5e71538a7610ce7b181ace904f4986fd96",
        urls = ["https://github.com/gimletlabs/mediapipe/archive/ce4f6d5e71538a7610ce7b181ace904f4986fd96.tar.gz"],
    ),
    com_nvidia_jetson_multimedia_api = dict(
        sha256 = "a28d46509bbe2c2f0dd40c9e43854b5cf95c33ac04502176c68f3c2f1cd7883e",
        urls = ["https://repo.download.nvidia.com/jetson/common/pool/main/n/nvidia-l4t-jetson-multimedia-api/nvidia-l4t-jetson-multimedia-api_35.4.1-20230801124926_arm64.deb"],
    ),
    com_nvidia_jetson_multimedia_utils = dict(
        sha256 = "f390756fa416f13285ec9647499baa3d9eaad262b9932257aafdff46f56c9580",
        urls = ["https://repo.download.nvidia.com/jetson/t194/pool/main/n/nvidia-l4t-multimedia-utils/nvidia-l4t-multimedia-utils_35.4.1-20230801124926_arm64.deb"],
    ),
    com_nvidia_l4t_camera = dict(
        sha256 = "d03d2f5baa111681aa3115e40e24572fba562e88b499a27f525ccf751fb36701",
        urls = ["https://repo.download.nvidia.com/jetson/t234/pool/main/n/nvidia-l4t-camera/nvidia-l4t-camera_35.4.1-20230801124926_arm64.deb"],
    ),
    com_github_cisco_openh264 = dict(
        sha256 = "453afa66dacb560bc5fd0468aabee90c483741571bca820a39a1c07f0362dc32",
        strip_prefix = "openh264-2.3.1",
        urls = ["https://github.com/cisco/openh264/archive/refs/tags/v2.3.1.tar.gz"],
    ),
    io_opentelemetry_cpp = dict(
        sha256 = "09c208a21fb1159d114a3ea15dc1bcc5dee28eb39907ba72a6012d2c7b7564a0",
        strip_prefix = "opentelemetry-cpp-1.12.0",
        urls = ["https://github.com/open-telemetry/opentelemetry-cpp/archive/refs/tags/v1.12.0.tar.gz"],
    ),
)
