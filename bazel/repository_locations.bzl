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
    aspect_rules_js = dict(
        sha256 = "77c4ea46c27f96e4aadcc580cd608369208422cf774988594ae8a01df6642c82",
        strip_prefix = "rules_js-1.32.2",
        urls = [
            "https://github.com/aspect-build/rules_js/releases/download/v1.32.2/rules_js-v1.32.2.tar.gz",
        ],
    ),
    aspect_rules_ts = dict(
        sha256 = "8aabb2055629a7becae2e77ae828950d3581d7fc3602fe0276e6e039b65092cb",
        strip_prefix = "rules_ts-2.0.0",
        urls = [
            "https://github.com/aspect-build/rules_ts/releases/download/v2.0.0/rules_ts-v2.0.0.tar.gz",
        ],
    ),
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
        sha256 = "738c139545e170daeeeb338879fac627648958f8ff07716d121de63b6599f3a6",
        strip_prefix = "rules_nodejs_gazelle-0.4.0",
        urls = [
            "https://github.com/benchsci/rules_nodejs_gazelle/archive/refs/tags/v0.4.0.tar.gz",
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
        sha256 = "a02a1e06b62ba462f9e70c73968f026d3c3e7daa4cbe967cc5a62b0778c8193b",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/clang-min-15.0-20230921235320.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_x86_64_glibc_host = dict(
        sha256 = "bd71e0fc496c10804840827161f183f2430304d2f272b622a3c101744c4799d3",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/llvm-15.0-20230921235320-libstdc%2B%2B.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host = dict(
        sha256 = "aa8678865a964919ac08c1f2df478990708b879aab7a8cd61f1d4855450e7878",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host_asan = dict(
        sha256 = "22cb1336eac935f9a002f1a5b923eff1a98fa689c71b105b335ab035395700f3",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-asan.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host_msan = dict(
        sha256 = "4a44fde041d24327aeaa185d8115b1057f06100fbdb6ff48e8d8ac6fcc158785",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-msan.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host_tsan = dict(
        sha256 = "645013c2509930d5f34dfa874e1c95799cd24503b72895864f7e3c049b4e4062",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-tsan.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_x86_64_glibc2_36 = dict(
        sha256 = "575cb6713705c0b32ba29aa0611995f83442ee7bf90e1aebc20440533b381ea4",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/llvm-15.0-20230921235320-libstdc++-x86_64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc2_36 = dict(
        sha256 = "1483829a2e033daf75bf8b13063ee00286c13dfba44ffc191371d342baba89d1",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-x86_64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_aarch64_glibc2_36 = dict(
        sha256 = "1483829a2e033daf75bf8b13063ee00286c13dfba44ffc191371d342baba89d1",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/llvm-15.0-20230921235320-libstdc++-aarch64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_aarch64_glibc2_36 = dict(
        sha256 = "d8583cbd55a66c6ea2d53268b3ae4829dae6d9e16e5d8040646f2cf1b7d8cddf",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-aarch64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_x86_64_glibc_host = dict(
        sha256 = "66a536aca79faa3c8143fbe2f035b9f352063c8008797b8f67da910e7242e2f3",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/libcxx-15.0-20230921235320.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_x86_64_glibc2_36 = dict(
        sha256 = "9ac0c76b214a1af0a2f3f3987381776041efb757b4707cd1a506c30b8fa2b629",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/libcxx-15.0-20230921235320-x86_64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_aarch64_glibc2_36 = dict(
        sha256 = "4881c10d4a09238db2c7dd4c1cd985a631fe66b3597a8a0499e5a665b4ea7f0e",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra-public/clang/15.0-20230921235320/libcxx-15.0-20230921235320-aarch64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    org_tensorflow = dict(
        sha256 = "99c732b92b1b37fc243a559e02f9aef5671771e272758aa4aec7f34dc92dac48",
        strip_prefix = "tensorflow-2.11.0",
        urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.11.0.tar.gz"],
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
)

PROTO_REPOSITORIES = dict(
    com_github_gogo_protobuf = dict(
        sha256 = "f89f8241af909ce3226562d135c25b28e656ae173337b3e58ede917aa26e1e3c",
        strip_prefix = "protobuf-1.3.2",
        urls = [
            "https://mirror.bazel.build/github.com/gogo/protobuf/archive/refs/tags/v1.3.2.zip",
            "https://github.com/gogo/protobuf/archive/refs/tags/v1.3.2.zip",
        ],
    ),
)
