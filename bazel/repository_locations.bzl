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
        sha256 = "79261f294085c5a5d86f1a2eba09219e3bbb2429d648e4fffd8e66e1051e843d",
        strip_prefix = "mediapipe-19261a6fbd2cdd4b5df79eeff3d26f544a013c0e",
        urls = ["https://github.com/gimletlabs/mediapipe/archive/19261a6fbd2cdd4b5df79eeff3d26f544a013c0e.tar.gz"],
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
    com_github_oneapi_oneTBB = dict(
        sha256 = "487023a955e5a3cc6d3a0d5f89179f9b6c0ae7222613a7185b0227ba0c83700b",
        strip_prefix = "oneTBB-2021.10.0",
        urls = ["https://github.com/oneapi-src/oneTBB/archive/refs/tags/v2021.10.0.tar.gz"],
    ),
    com_github_openvinotoolkit_openvino = dict(
        sha256 = "ff88596b342440185874ddbe22874b47ad7b923f14671921af760b15c98aacd6",
        strip_prefix = "openvino-2023.1.0",
        urls = ["https://github.com/openvinotoolkit/openvino/archive/refs/tags/2023.1.0.tar.gz"],
    ),
    com_github_openvinotoolkit_oneDNN = dict(
        sha256 = "3c51d577f9e7e4cbd94ad08d267502953ec64513241dda6595b2608fafc8314c",
        strip_prefix = "oneDNN-2ead5d4fe5993a797d9a7a4b8b5557b96f6ec90e",
        urls = ["https://github.com/openvinotoolkit/oneDNN/archive/2ead5d4fe5993a797d9a7a4b8b5557b96f6ec90e.tar.gz"],
    ),
    com_github_openvinotoolkit_mlas = dict(
        sha256 = "b7fdd19523a88373d19fd8d5380f64c2834040fa50a6f0774acf08f3fa858daa",
        strip_prefix = "mlas-7a35e48a723944972088627be1a8b60841e8f6a5",
        urls = ["https://github.com/openvinotoolkit/mlas/archive/7a35e48a723944972088627be1a8b60841e8f6a5.tar.gz"],
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
)

# To use a local repo for local development, add the path to point to your local directory below.
#   ex: path = "/home/user/path/to/repo"
# then replace `_bazel_repo(repo, ...)` with `_local_repo(repo, ...)` in `repositories.bzl`.
LOCAL_REPOSITORY_LOCATIONS = dict(
    com_example_repo = dict(
        path = "/home/user/path/to_repo",
    ),
)
