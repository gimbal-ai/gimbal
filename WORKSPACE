workspace(name = "gml")

load("//:workspace.bzl", "check_min_bazel_version")

check_min_bazel_version("6.3.2")

load("//bazel:repositories.bzl", "gml_cc_toolchain_deps", "gml_deps")

gml_cc_toolchain_deps()

load("//bazel/cc_toolchains/sysroots:create_sysroots.bzl", "create_sysroots")

create_sysroots()

load("//bazel/cc_toolchains/sysroots:register_sysroots.bzl", "register_sysroots")

register_sysroots()

load("//bazel/cc_toolchains:create_toolchains.bzl", "gml_create_cc_toolchains")

gml_create_cc_toolchains()

load("//bazel/cc_toolchains:register_toolchains.bzl", "gml_register_cc_toolchains")

gml_register_cc_toolchains()

gml_deps()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

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
