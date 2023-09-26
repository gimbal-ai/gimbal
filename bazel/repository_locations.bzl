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
    aspect_bazel_lib = dict(
        sha256 = "09b51a9957adc56c905a2c980d6eb06f04beb1d85c665b467f659871403cf423",
        strip_prefix = "bazel-lib-1.34.5",
        urls = [
            "https://github.com/aspect-build/bazel-lib/releases/download/v1.34.5/bazel-lib-v1.34.5.tar.gz",
        ],
    ),
    aspect_rules_jest = dict(
        sha256 = "098186ffc450f2a604843d8ba14217088a0e259ea6a03294af5360a7f1bcd3e8",
        strip_prefix = "rules_jest-0.19.5",
        urls = [
            "https://github.com/aspect-build/rules_jest/releases/download/v0.19.5/rules_jest-v0.19.5.tar.gz",
        ],
    ),
    bazel_gazelle = dict(
        sha256 = "d3fa66a39028e97d76f9e2db8f1b0c11c099e8e01bf363a923074784e451f809",
        urls = [
            "https://github.com/bazelbuild/bazel-gazelle/releases/download/v0.33.0/bazel-gazelle-v0.33.0.tar.gz",
        ],
    ),
    rules_pkg = dict(
        sha256 = "8f9ee2dc10c1ae514ee599a8b42ed99fa262b757058f65ad3c384289ff70c4b8",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
            "https://github.com/bazelbuild/rules_pkg/releases/download/0.9.1/rules_pkg-0.9.1.tar.gz",
        ],
    ),
    bazel_skylib = dict(
        sha256 = "f7be3474d42aae265405a592bb7da8e171919d74c16f082a5457840f06054728",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/1.2.1/bazel-skylib-1.2.1.tar.gz",
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
    com_github_bazelbuild_buildtools = dict(
        sha256 = "42968f9134ba2c75c03bb271bd7bb062afb7da449f9b913c96e5be4ce890030a",
        strip_prefix = "buildtools-6.3.3",
        urls = ["https://github.com/bazelbuild/buildtools/archive/refs/tags/v6.3.3.tar.gz"],
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
    com_github_google_glog = dict(
        sha256 = "95dc9dd17aca4e12e2cb18087a5851001f997682f5f0d0c441a5be3b86f285bd",
        strip_prefix = "glog-bc1fada1cf63ad12aee26847ab9ed4c62cffdcf9",
        # We cannot use the last released version due to https://github.com/google/glog/pull/706
        # Once there is a realease that includes that fix, we can switch to a released version.
        urls = ["https://github.com/google/glog/archive/bc1fada1cf63ad12aee26847ab9ed4c62cffdcf9.tar.gz"],
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
    com_google_absl = dict(
        sha256 = "91ac87d30cc6d79f9ab974c51874a704de9c2647c40f6932597329a282217ba8",
        strip_prefix = "abseil-cpp-20220623.1",
        urls = ["https://github.com/abseil/abseil-cpp/archive/refs/tags/20220623.1.tar.gz"],
    ),
    com_google_benchmark = dict(
        sha256 = "3aff99169fa8bdee356eaa1f691e835a6e57b1efeadb8a0f9f228531158246ac",
        strip_prefix = "benchmark-1.7.0",
        urls = ["https://github.com/google/benchmark/archive/refs/tags/v1.7.0.tar.gz"],
    ),
    com_google_flatbuffers = dict(
        sha256 = "e2dc24985a85b278dd06313481a9ca051d048f9474e0f199e372fea3ea4248c9",
        strip_prefix = "flatbuffers-2.0.6",
        urls = ["https://github.com/google/flatbuffers/archive/refs/tags/v2.0.6.tar.gz"],
    ),
    com_google_googletest = dict(
        sha256 = "81964fe578e9bd7c94dfdb09c8e4d6e6759e19967e397dbea48d1c10e45d0df2",
        strip_prefix = "googletest-release-1.12.1",
        urls = ["https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz"],
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
    com_googlesource_code_re2 = dict(
        urls = ["https://github.com/google/re2/archive/refs/tags/2021-08-01.tar.gz"],
        strip_prefix = "re2-2021-08-01",
        sha256 = "cd8c950b528f413e02c12970dce62a7b6f37733d7f68807e73a2d9bc9db79bc8",
    ),
    com_intel_tbb = dict(
        sha256 = "91eab849ab1442db72317f8c968c5a1010f8546ca35f26086201262096c8a8a9",
        strip_prefix = "oneTBB-e6104c9599f7f10473caf545199f7468c0a8e52f",
        urls = ["https://github.com/oneapi-src/oneTBB/archive/e6104c9599f7f10473caf545199f7468c0a8e52f.tar.gz"],
    ),
    com_llvm_clang_15 = dict(
        sha256 = "a02a1e06b62ba462f9e70c73968f026d3c3e7daa4cbe967cc5a62b0778c8193b",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/clang-min-15.0-20230921235320.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_x86_64_glibc_host = dict(
        sha256 = "bd71e0fc496c10804840827161f183f2430304d2f272b622a3c101744c4799d3",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/llvm-15.0-20230921235320-libstdc%2B%2B.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host = dict(
        sha256 = "aa8678865a964919ac08c1f2df478990708b879aab7a8cd61f1d4855450e7878",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host_asan = dict(
        sha256 = "22cb1336eac935f9a002f1a5b923eff1a98fa689c71b105b335ab035395700f3",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-asan.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host_msan = dict(
        sha256 = "4a44fde041d24327aeaa185d8115b1057f06100fbdb6ff48e8d8ac6fcc158785",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-msan.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc_host_tsan = dict(
        sha256 = "645013c2509930d5f34dfa874e1c95799cd24503b72895864f7e3c049b4e4062",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-tsan.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_x86_64_glibc2_36 = dict(
        sha256 = "575cb6713705c0b32ba29aa0611995f83442ee7bf90e1aebc20440533b381ea4",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/llvm-15.0-20230921235320-libstdc++-x86_64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_x86_64_glibc2_36 = dict(
        sha256 = "1483829a2e033daf75bf8b13063ee00286c13dfba44ffc191371d342baba89d1",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-x86_64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_aarch64_glibc2_36 = dict(
        sha256 = "1483829a2e033daf75bf8b13063ee00286c13dfba44ffc191371d342baba89d1",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/llvm-15.0-20230921235320-libstdc++-aarch64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_lib_libcpp_aarch64_glibc2_36 = dict(
        sha256 = "d8583cbd55a66c6ea2d53268b3ae4829dae6d9e16e5d8040646f2cf1b7d8cddf",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/llvm-15.0-20230921235320-libcxx-aarch64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_x86_64_glibc_host = dict(
        sha256 = "66a536aca79faa3c8143fbe2f035b9f352063c8008797b8f67da910e7242e2f3",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/libcxx-15.0-20230921235320.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_x86_64_glibc2_36 = dict(
        sha256 = "9ac0c76b214a1af0a2f3f3987381776041efb757b4707cd1a506c30b8fa2b629",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/libcxx-15.0-20230921235320-x86_64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    com_llvm_libcxx_aarch64_glibc2_36 = dict(
        sha256 = "4881c10d4a09238db2c7dd4c1cd985a631fe66b3597a8a0499e5a665b4ea7f0e",
        strip_prefix = "",
        urls = [
            "https://storage.googleapis.com/gimlet-dev-infra/clang/15.0-20230921235320/libcxx-15.0-20230921235320-aarch64-sysroot.tar.gz",
        ],
        manual_license_name = "llvm/llvm-project",
    ),
    io_bazel_rules_go = dict(
        sha256 = "278b7ff5a826f3dc10f04feaf0b70d48b68748ccd512d7f98bf442077f043fe3",
        urls = [
            "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.41.0/rules_go-v0.41.0.zip",
            "https://github.com/bazelbuild/rules_go/releases/download/v0.41.0/rules_go-v0.41.0.zip",
        ],
    ),
    org_tensorflow = dict(
        sha256 = "99c732b92b1b37fc243a559e02f9aef5671771e272758aa4aec7f34dc92dac48",
        strip_prefix = "tensorflow-2.11.0",
        urls = ["https://github.com/tensorflow/tensorflow/archive/refs/tags/v2.11.0.tar.gz"],
    ),
    rules_foreign_cc = dict(
        sha256 = "6041f1374ff32ba711564374ad8e007aef77f71561a7ce784123b9b4b88614fc",
        strip_prefix = "rules_foreign_cc-0.8.0",
        urls = ["https://github.com/bazelbuild/rules_foreign_cc/archive/refs/tags/0.8.0.tar.gz"],
    ),
    rules_python = dict(
        sha256 = "cdf6b84084aad8f10bf20b46b77cb48d83c319ebe6458a18e9d2cebf57807cdd",
        strip_prefix = "rules_python-0.8.1",
        urls = ["https://github.com/bazelbuild/rules_python/archive/refs/tags/0.8.1.tar.gz"],
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
)
