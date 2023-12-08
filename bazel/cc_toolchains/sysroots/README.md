# Sysroots

## Add a package to a sysroot

1. Add the package to the relevant packages spec in `bazel/cc_toolchains/sysroots/packages`.
1. Run `bazel run //bazel/cc_toolchains/sysroots/packages:all_debs.update`
1. Add the package to the sysroot definition in `bazel/cc_toolchains/sysroots/sysroots.bzl`.

For example, if you wanted to add the package `chromium` (<https://packages.debian.org/bookworm/chromium>) to the default test sysroot, you would do the following:

1. Add the line `- name: chromium` to `bazel/cc_toolchains/sysroots/packages/debian12.yaml` under the key `packages`.
1. Run `bazel run //bazel/cc_toolchains/sysroots/packages:all_debs.update`.
1. Add `"debian12_chromium"` to the `test_pkgs` list inside the macro `_debian12_sysroots` in `bazel/cc_toolchains/sysroots/sysroots.bzl`

## Add a new sysroot

1. If the new sysroot uses the same package repositories as an existing sysroot (eg. debian bookworm packages, or nvidia jetson packages), you can skip to step 4.
1. Otherwise, create a new yaml file in `bazel/cc_toolchains/sysroots/packages`. Copy the structure of an existing yaml file in that directory. The important fields are `repositories` and `packages`, the others are optional.
1. Run `bazel run //bazel/cc_toolchains/sysroots/packages:all_debs.update`
1. Add a new macro in `bazel/cc_toolchains/sysroots/sysroots.bzl` with the `sysroot_repo` call for your sysroot. See the existing macros for examples of calling `sysroot_repo`. (If you want the sysroot to have more than one of the `runtime`, `build` and `test` variants, you should define them all in the same macro).
1. Call your new macro in `_gml_sysroots` in the same file.
1. Add a new value in the `sysroot_type` flag inside of `bazel/cc_toolchains/sysroots/BUILD.bazel` with the name of your new type of sysroot.
1. In the same build file add a new `config_setting` corresponding with the new `sysroot_type` you just added.
1. If you defined a `build` variant sysroot, you need to add a new `clang_toolchain` in `bazel/cc_toolchains/toolchains.bzl`.
1. For any variant you didn't define, you need to add your new `config_setting` to the `match_any` attribute of the corresponding `config_setting_group` in `bazel/cc_toolchains/sysroots/BUILD.bazel`.
For example, if you didn't define a `build` sysroot, you should add your `config_setting` target to the `match_any` attribute of the `use_debian12_build_sysroot` `config_setting_group`.
1. Now you should be able to use your new sysroot by running bazel with `--//bazel/cc_toolchains/sysroots:sysroot_type=<type added from step 6>`
