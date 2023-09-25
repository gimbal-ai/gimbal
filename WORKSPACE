workspace(name = "gml")

load("//bazel:repositories.bzl", "gml_deps")

gml_deps()

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "278b7ff5a826f3dc10f04feaf0b70d48b68748ccd512d7f98bf442077f043fe3",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/rules_go/releases/download/v0.41.0/rules_go-v0.41.0.zip",
        "https://github.com/bazelbuild/rules_go/releases/download/v0.41.0/rules_go-v0.41.0.zip",
    ],
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(version = "1.21.1")

# gazelle:repo bazel_gazelle
load("@bazel_gazelle//:deps.bzl", "gazelle_dependencies")
load("//:go_deps.bzl", "gml_go_dependencies")

# gazelle:repository_macro go_deps.bzl%gml_go_dependencies
gml_go_dependencies()

gazelle_dependencies(go_sdk = "go_sdk")

# Setup rules_js and related tooling
load("@aspect_rules_js//js:repositories.bzl", "rules_js_dependencies")
rules_js_dependencies()

load("@aspect_rules_ts//ts:repositories.bzl", "rules_ts_dependencies", "LATEST_TYPESCRIPT_VERSION")
rules_ts_dependencies(ts_version = LATEST_TYPESCRIPT_VERSION)

load("@aspect_rules_jest//jest:dependencies.bzl", "rules_jest_dependencies")
rules_jest_dependencies()

load("@rules_nodejs//nodejs:repositories.bzl", "DEFAULT_NODE_VERSION", "nodejs_register_toolchains")
nodejs_register_toolchains(
    name = "nodejs",
    node_version = DEFAULT_NODE_VERSION,
)
