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
        sha256 = "c3734c1d4f18f58c74e1efb1ab83dd2bed84d0de2e0b26c8c0fcb649bdbb75a1",
        strip_prefix = "rules_nodejs_gazelle-0.5.0",
        urls = [
            "https://github.com/benchsci/rules_nodejs_gazelle/archive/refs/tags/v0.5.0.tar.gz",
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
        sha256 = "a22461d13119ac5c78f205d3df1db13403e58ce1bb1794edc9313677313f4a9d",
        urls = ["https://github.com/nlohmann/json/releases/download/v3.11.3/include.zip"],
    ),
    com_github_okdshin_picosha = dict(
        sha256 = "18d82bb79c021ccf4ce58125b64691accef54237ba5194462740bacf8b39d8a9",
        strip_prefix = "PicoSHA2-27fcf6979298949e8a462e16d09a0351c18fcaf2",
        urls = ["https://github.com/okdshin/PicoSHA2/archive/27fcf6979298949e8a462e16d09a0351c18fcaf2.tar.gz"],
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
        sha256 = "b0dac36faf6d8da9dbadccb6c058716214562e1b100fbc829d96f89fd5a7d99a",
        strip_prefix = "tensorflow-49e79ddefccfbce482bd85db1628f8fc50e5f912",
        urls = ["https://github.com/gimletlabs/tensorflow/archive/49e79ddefccfbce482bd85db1628f8fc50e5f912.tar.gz"],
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
        sha256 = "ee7a11d66e7bbc5b0f7a35ca3e960cb9a5f8a314b22252e19912dfbc6e22782d",
        strip_prefix = "rules_proto-3.1.0",
        urls = [
            "https://github.com/stackb/rules_proto/archive/v3.1.0.tar.gz",
        ],
    ),
    com_github_ffmpeg_nv_codec_headers = dict(
        sha256 = "62b30ab37e4e9be0d0c5b37b8fee4b094e38e570984d56e1135a6b6c2c164c9f",
        strip_prefix = "nv-codec-headers-12.1.14.0",
        urls = [
            "https://github.com/FFmpeg/nv-codec-headers/releases/download/n12.1.14.0/nv-codec-headers-12.1.14.0.tar.gz",
        ],
    ),
    com_github_ffmpeg_ffmpeg = dict(
        sha256 = "7c1ebea95d815e49c1e60c7ee816410dec73a81b8ac002b276780d2f9048e598",
        strip_prefix = "FFmpeg-n6.1.1",
        urls = [
            "https://github.com/FFmpeg/FFmpeg/archive/refs/tags/n6.1.1.tar.gz",
        ],
    ),
    com_github_opencv_opencv = dict(
        sha256 = "62f650467a60a38794d681ae7e66e3e8cfba38f445e0bf87867e2f2cdc8be9d5",
        strip_prefix = "opencv-4.8.1",
        urls = ["https://github.com/opencv/opencv/archive/refs/tags/4.8.1.tar.gz"],
    ),
    com_github_opencv_contrib = dict(
        sha256 = "0c082a0b29b3118f2a0a1856b403bb098643af7b994a0080f402a12159a99c6e",
        strip_prefix = "opencv_contrib-4.8.1",
        urls = ["https://github.com/opencv/opencv_contrib/archive/refs/tags/4.8.1.tar.gz"],
    ),
    cpuinfo = dict(
        sha256 = "109e9d2f95a0d72fe50174b44de5b8c9be3e9551407b882e4ad497e9e068d202",
        strip_prefix = "cpuinfo-87d8234510367db49a65535021af5e1838a65ac2",
        urls = ["https://github.com/pytorch/cpuinfo/archive/87d8234510367db49a65535021af5e1838a65ac2.tar.gz"],
    ),
    XNNPACK = dict(
        sha256 = "4c85e07f3bc5602266e490ff9cea4f29daef9865ef57cf561d3f32fbf9ee51fc",
        strip_prefix = "XNNPACK-7f3e8aa632ab976b8a195c8d3d17e2f5831dde0e",
        urls = ["https://github.com/google/XNNPACK/archive/7f3e8aa632ab976b8a195c8d3d17e2f5831dde0e.tar.gz"],
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
        sha256 = "85403c8f67bb79bce18d1ddc79ceebdbc33f2c780a3a3e8f1f7dbcf4b33cc336",
        strip_prefix = "mediapipe-041f016f96ac99cea9e24f1b9fd56be774c5ab40",
        urls = ["https://github.com/gimletlabs/mediapipe/archive/041f016f96ac99cea9e24f1b9fd56be774c5ab40.tar.gz"],
    ),
    com_github_cisco_openh264 = dict(
        sha256 = "453afa66dacb560bc5fd0468aabee90c483741571bca820a39a1c07f0362dc32",
        strip_prefix = "openh264-2.3.1",
        urls = ["https://github.com/cisco/openh264/archive/refs/tags/v2.3.1.tar.gz"],
    ),
    io_opentelemetry_cpp = dict(
        sha256 = "c406ee37be1e0411c07b1c2cfde19aee5e85ca308846ae2d596b1d89cb0f0105",
        strip_prefix = "opentelemetry-cpp-61b676d63a8b50b1ceca61d717b7f8a699f24b34",
        urls = ["https://github.com/gimletlabs/opentelemetry-cpp/archive/61b676d63a8b50b1ceca61d717b7f8a699f24b34.tar.gz"],
    ),
    com_github_oneapi_oneTBB = dict(
        sha256 = "487023a955e5a3cc6d3a0d5f89179f9b6c0ae7222613a7185b0227ba0c83700b",
        strip_prefix = "oneTBB-2021.10.0",
        urls = ["https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.10.0.tar.gz"],
    ),
    com_github_openvinotoolkit_openvino = dict(
        sha256 = "53fccad05279d0975eca84ec75517a7c360be9b0f7bcd822da29a7949c12ce70",
        strip_prefix = "openvino-2024.3.0",
        urls = ["https://github.com/openvinotoolkit/openvino/archive/refs/tags/2024.3.0.tar.gz"],
    ),
    com_github_openvinotoolkit_oneDNN = dict(
        sha256 = "13bee5b8522177f297e095e3eba5948c1a7ee7a816d19d5a59ce0f717f82cedc",
        strip_prefix = "oneDNN-f0f8defe2dff5058391f2a66e775e20b5de33b08",
        urls = ["https://github.com/openvinotoolkit/oneDNN/archive/f0f8defe2dff5058391f2a66e775e20b5de33b08.tar.gz"],
    ),
    com_github_openvinotoolkit_mlas = dict(
        sha256 = "0a44fbfd4b13e8609d66ddac4b11a27c90c1074cde5244c91ad197901666004c",
        strip_prefix = "mlas-d1bc25ec4660cddd87804fcf03b2411b5dfb2e94",
        urls = ["https://github.com/openvinotoolkit/mlas/archive/d1bc25ec4660cddd87804fcf03b2411b5dfb2e94.tar.gz"],
    ),
    com_github_zeux_pugixml = dict(
        sha256 = "2f10e276870c64b1db6809050a75e11a897a8d7456c4be5c6b2e35a11168a015",
        strip_prefix = "pugixml-1.14",
        urls = ["https://github.com/zeux/pugixml/releases/download/v1.14/pugixml-1.14.tar.gz"],
    ),
    com_github_herumi_xbyak = dict(
        sha256 = "41f3dc7727a48c751024c92fa4da24a4a1e0ed16b7930c79d05b76960b19562d",
        strip_prefix = "xbyak-6.73",
        urls = ["https://github.com/herumi/xbyak/archive/refs/tags/v6.73.tar.gz"],
    ),
    com_github_onnx_onnx = dict(
        sha256 = "c757132e018dd0dd171499ef74fca88b74c5430a20781ec53da19eb7f937ef68",
        strip_prefix = "onnx-1.15.0",
        urls = ["https://github.com/onnx/onnx/archive/refs/tags/v1.15.0.tar.gz"],
    ),
    com_github_khronosgroup_opencl_headers = dict(
        sha256 = "0ce992f4167f958f68a37918dec6325be18f848dee29a4521c633aae3304915d",
        strip_prefix = "OpenCL-Headers-2023.04.17",
        urls = ["https://github.com/KhronosGroup/OpenCL-Headers/archive/refs/tags/v2023.04.17.tar.gz"],
    ),
    com_github_khronosgroup_opencl_icd_loader = dict(
        sha256 = "173bdc4f321d550b6578ad2aafc2832f25fbb36041f095e6221025f74134b876",
        strip_prefix = "OpenCL-ICD-Loader-2023.04.17",
        urls = ["https://github.com/KhronosGroup/OpenCL-ICD-Loader/archive/refs/tags/v2023.04.17.tar.gz"],
    ),
    com_github_khronosgroup_opencl_clhpp = dict(
        sha256 = "179243843c620ef6f78b52937aaaa0a742c6ff415f9aaefe3c20225ee283b357",
        strip_prefix = "OpenCL-CLHPP-2023.04.17",
        urls = ["https://github.com/KhronosGroup/OpenCL-CLHPP/archive/refs/tags/v2023.04.17.tar.gz"],
    ),
    com_github_googlecoral_libedgetpu = dict(
        sha256 = "86a6e654e093c204b4fb579a60773bfa080f095c9cbb3a2c114ca4a13e0b15eb",
        strip_prefix = "libedgetpu-release-grouper",
        urls = ["https://github.com/google-coral/libedgetpu/archive/refs/tags/release-grouper.tar.gz"],
    ),
    com_github_oneapi_level_zero = dict(
        sha256 = "f341dd6355d8da6ee9c29031642b8e8e4259f91c13c72d318c81663af048817e",
        strip_prefix = "level-zero-1.16.1",
        urls = ["https://github.com/oneapi-src/level-zero/archive/refs/tags/v1.16.1.tar.gz"],
    ),
    com_github_llvm_llvm_project = dict(
        sha256 = "cb6f82a8413bd518499c3081a5c396f8a140a7c650201c201892cb6ce4f4f397",
        strip_prefix = "llvm-project-676d3bafc09d0c331a04b813804407334de12917",
        urls = ["https://github.com/llvm/llvm-project/archive/676d3bafc09d0c331a04b813804407334de12917.tar.gz"],
        manual_license_name = "llvm/llvm-project",
    ),
    llvm_zstd = dict(
        sha256 = "7c42d56fac126929a6a85dbc73ff1db2411d04f104fae9bdea51305663a83fd0",
        strip_prefix = "zstd-1.5.2",
        urls = ["https://github.com/facebook/zstd/releases/download/v1.5.2/zstd-1.5.2.tar.gz"],
    ),
    llvm_zlib = dict(
        sha256 = "e36bb346c00472a1f9ff2a0a4643e590a254be6379da7cddd9daeb9a7f296731",
        strip_prefix = "zlib-ng-2.0.7",
        urls = ["https://github.com/zlib-ng/zlib-ng/archive/refs/tags/2.0.7.zip"],
    ),
    com_github_llvm_torch_mlir = dict(
        sha256 = "f66ea17d4462bf394519c25abf3fe67729919b0df311ee4d8dacf9eb06e35e05",
        strip_prefix = "torch-mlir-4549ca1e2379f5a5c1cde22e0307c4263cbee262",
        urls = ["https://github.com/gimletlabs/torch-mlir/archive/4549ca1e2379f5a5c1cde22e0307c4263cbee262.tar.gz"],
    ),
    com_github_openxla_stablehlo = dict(
        sha256 = "0d567e9a5f0f6c8487272dc27539403f1c9cae5039e64c52a648ba08ae11f86d",
        strip_prefix = "stablehlo-271e8634de184fbfafd677d3876170feb6d08c97",
        urls = ["https://github.com/openxla/stablehlo/archive/271e8634de184fbfafd677d3876170feb6d08c97.tar.gz"],
    ),
    com_github_vertical_beach_bytetrack_cpp = dict(
        sha256 = "21df68687576756ece52f71b466c651833fa16fee1ab34114f2616475ddb7576",
        strip_prefix = "ByteTrack-cpp-22e2ebf217ad43d0d1f8382195828afd3fee1e43",
        urls = ["https://github.com/gimletlabs/ByteTrack-cpp/archive/22e2ebf217ad43d0d1f8382195828afd3fee1e43.tar.gz"],
    ),
)

# To use a local repo for local development, simply add a `local_path` key to the relevant
# repository location in `REPOSITORY_LOCATIONS`.
#     e.g. local_path = "/home/user/path/to/repo
# The local repo will take precedence over the URL location.
#
# Note: Patches in the corresponding _bazel_repo in `repositories.bzl` are not supported,
#       so those should be removed to avoid errors.
