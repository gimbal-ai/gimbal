workspace(name = "gml")

load("//:workspace.bzl", "check_min_bazel_version")

check_min_bazel_version("6.3.2")

load("//bazel:repositories.bzl", "gml_cc_toolchain_deps", "gml_deps", "llvm_deps")

gml_cc_toolchain_deps()

load("//bazel/cc_toolchains/sysroots/packages:all_debs.bzl", "deb_repos")

deb_repos()

load("//bazel/cc_toolchains/sysroots:sysroots.bzl", "gml_sysroots")

gml_sysroots()

load("//bazel/cc_toolchains:toolchains.bzl", "gml_cc_toolchains")

gml_cc_toolchains()

gml_deps()

llvm_deps()

load("@bazel_skylib//:workspace.bzl", "bazel_skylib_workspace")

bazel_skylib_workspace()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@io_opentelemetry_cpp//bazel:repository.bzl", "opentelemetry_cpp_deps")

opentelemetry_cpp_deps()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

# These dependencies are needed by GRPC.
load("@build_bazel_rules_apple//apple:repositories.bzl", "apple_rules_dependencies")

apple_rules_dependencies()

load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")

apple_support_dependencies()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

load("//bazel:gogo.bzl", "gogo_grpc_proto")

gogo_grpc_proto(name = "gogo_grpc_proto")

load("@build_stack_rules_proto//deps:core_deps.bzl", "core_deps")

core_deps()

load("@build_stack_rules_proto//:go_deps.bzl", "gazelle_protobuf_extension_go_deps")

gazelle_protobuf_extension_go_deps()

load("@build_stack_rules_proto//deps:go_core_deps.bzl", rules_proto_go_core_deps = "go_core_deps")

rules_proto_go_core_deps()

load("@build_stack_rules_proto//deps:protobuf_core_deps.bzl", "protobuf_core_deps")

protobuf_core_deps()

register_toolchains("@build_stack_rules_proto//toolchain:standard")

load("@llvm-raw//utils/bazel:configure.bzl", "llvm_configure")

llvm_configure(
    name = "llvm-project",
    repo_mapping = {
        "@python_runtime": "@python",
    },
)

load("@torch-mlir-raw//utils/bazel:configure.bzl", "torch_mlir_configure")

torch_mlir_configure(
    name = "torch-mlir",
)

# mediapipe dependencies
load("@org_tensorflow//tensorflow:workspace3.bzl", "tf_workspace3")

tf_workspace3()

load("@org_tensorflow//tensorflow:workspace2.bzl", "tf_workspace2")

tf_workspace2()

load("@org_tensorflow//tensorflow:workspace1.bzl", "tf_workspace1")

tf_workspace1(with_rules_cc = False)

load("@org_tensorflow//tensorflow:workspace0.bzl", "tf_workspace0")

tf_workspace0()

load("@com_github_googlecoral_libedgetpu//:workspace.bzl", "libedgetpu_dependencies")

libedgetpu_dependencies()

load("//bazel/external:mediapipe_repos.bzl", "mediapipe_repos")

mediapipe_repos()

load("//bazel/external:mediapipe_deps.bzl", "mediapipe_deps")

mediapipe_deps()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# cxxbridge-cmd is a binary only Rust crate, so we follow these instructions for it
# http://bazelbuild.github.io/rules_rust/crate_universe.html#binary-dependencies
http_archive(
    name = "cxxbridge_cmd",
    build_file = "//bazel/external/cxxbridge_cmd:BUILD.bazel.gml",
    sha256 = "b20c2a31fb5a2e2aeebced19e35d78a974301171391b39e36d5ebb00bea2aa93",
    strip_prefix = "cxxbridge-cmd-1.0.94",
    type = "tar.gz",
    urls = [
        "https://crates.io/api/v1/crates/cxxbridge-cmd/1.0.94/download",
    ],
)
