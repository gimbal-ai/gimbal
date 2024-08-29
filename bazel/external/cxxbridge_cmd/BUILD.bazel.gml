load("@rules_rust//rust:defs.bzl", "rust_binary")
load("@crates//:defs.bzl", "all_crate_deps")

rust_binary(
    name = "cxxbridge",
    srcs = glob(["src/**/*.rs"]),
    data = ["src/gen/include/cxx.h"],
    edition = "2018",
      deps = [
        "@crates//:clap",
        "@crates//:codespan-reporting",
        "@crates//:proc-macro2",
        "@crates//:quote",
        "@crates//:syn",
    ],
    rustc_flags = ["-Awarnings"],
    visibility = ["//visibility:public"],
)
