workspace(name = "gml")

load("//:workspace.bzl", "check_min_bazel_version")

check_min_bazel_version("6.3.2")

load("//bazel:repositories.bzl", "gml_cc_toolchain_deps", "gml_deps")

gml_cc_toolchain_deps()

load("//bazel/cc_toolchains/sysroots:sysroots.bzl", "gml_sysroot_deps")

gml_sysroot_deps()

load("//bazel/cc_toolchains:toolchains.bzl", "gml_register_cc_toolchains")

gml_register_cc_toolchains()

gml_deps()

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_register_toolchains(version = "1.21.1")

go_rules_dependencies()

# gazelle:repo bazel_gazelle
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")
load("//:go_deps.bzl", "gml_go_dependencies")

# gazelle:repository_macro go_deps.bzl%gml_go_dependencies
gml_go_dependencies()

load("@com_google_protobuf//:protobuf_deps.bzl", "protobuf_deps")

protobuf_deps()

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

# These dependencies are needed by GRPC.
load("@build_bazel_rules_apple//apple:repositories.bzl", "apple_rules_dependencies")

apple_rules_dependencies()

load("@build_bazel_apple_support//lib:repositories.bzl", "apple_support_dependencies")

apple_support_dependencies()

gazelle_dependencies(go_sdk = "go_sdk")

load("@aspect_bazel_lib//lib:repositories.bzl", "aspect_bazel_lib_dependencies")

aspect_bazel_lib_dependencies()

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

load("//bazel:proto_repositories.bzl", "gml_proto_deps")

gml_proto_deps()

register_toolchains("@build_stack_rules_proto//toolchain:standard")

# Setup rules_js and related tooling
load("@aspect_rules_js//js:repositories.bzl", "rules_js_dependencies")

rules_js_dependencies()

load("@aspect_rules_ts//ts:repositories.bzl", "LATEST_TYPESCRIPT_VERSION", "rules_ts_dependencies")

rules_ts_dependencies(ts_version = LATEST_TYPESCRIPT_VERSION)

load("@aspect_rules_jest//jest:dependencies.bzl", "rules_jest_dependencies")

rules_jest_dependencies()

load("@rules_nodejs//nodejs:repositories.bzl", "DEFAULT_NODE_VERSION", "nodejs_register_toolchains")

nodejs_register_toolchains(
    name = "nodejs",
    node_version = DEFAULT_NODE_VERSION,
)

load("@aspect_rules_js//npm:npm_import.bzl", "npm_translate_lock")

npm_translate_lock(
    name = "npm",
    bins = {
        # derived from "bin" attribute in node_modules/next/package.json
        "next": {
            "next": "./dist/bin/next",
        },
    },
    npmrc = "//src/ui:.npmrc",
    pnpm_lock = "//src/ui:pnpm-lock.yaml",
    verify_node_modules_ignored = "//:.bazelignore",
)

load("@npm//:repositories.bzl", "npm_repositories")

npm_repositories()
