# Copyright 2023- Gimlet Labs, Inc.
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

load("//bazel/cc_toolchains/sysroots:sysroot_repo.bzl", "sysroot_repo")

_DEFAULT_BUILD_PATH_PREFIXES = [
    "usr/local",
    "usr/include",
    "usr/lib/gcc",
    "lib",
    "lib64",
    "usr/lib",
]

# debian12 packages are used as the base for other sysroots that don't provide their own base packages.
# So we need to specify them globally.
_DEBIAN12_RUNTIME_PKGS = [
    "debian12_ca-certificates",
    "debian12_libtinfo6",
    "debian12_libc6",
    "debian12_libelf1",
    "debian12_libstdc++6",
    "debian12_zlib1g",
    "debian12_libunwind8",
    "debian12_libgles2-mesa",
    "debian12_libegl1-mesa",
]

_DEBIAN12_BUILD_PKGS = [
    "debian12_liblzma-dev",
    "debian12_libunwind-dev",
    "debian12_libncurses-dev",
    "debian12_libc6-dev",
    "debian12_libegl1-mesa-dev",
    "debian12_libelf-dev",
    "debian12_libgcc-12-dev",
    "debian12_libgles2-mesa-dev",
    "debian12_libicu-dev",
    "debian12_libstdc++-12-dev",
    "debian12_linux-libc-dev",
    "debian12_mesa-common-dev",
    "debian12_zlib1g-dev",
    "debian12_libva-dev",
]

_DEBIAN12_TEST_PKGS = [
    # dash provides /bin/sh. Things like popen will fail with weird errors if /bin/sh doesn't exist.
    "debian12_dash",
    # Some of our tests require these shell utilities
    "debian12_bash",
    "debian12_grep",
    "debian12_gawk",
    "debian12_sed",
    "debian12_libc-bin",
    "debian12_libmagic1",
    # Deps for podman
    "debian12_iptables",
    "debian12_aardvark-dns",
    "debian12_netavark",
    "debian12_podman",
    # Useful for "debug" containers that want to do basic file manipulation
    "debian12_coreutils",
    "debian12_tar",
    # Include python in the test sysroot.
    "debian12_python3.11",
]

def _debian12_with_extras(
        prefix,
        runtime_target_settings,
        build_target_settings,
        test_target_settings,
        extra_runtime_packages = [],
        extra_build_packages = [],
        extra_test_packages = [],
        extra_build_compile_flags = [],
        extra_build_link_flags = [],
        libc_version = "glibc2_36",
        supported_archs = ["aarch64", "x86_64"]):
    sysroot_repo(
        name = "{}_runtime".format(prefix),
        libc_version = libc_version,
        supported_archs = supported_archs,
        variant = "runtime",
        packages = _DEBIAN12_RUNTIME_PKGS + extra_runtime_packages,
        target_settings = runtime_target_settings,
    )

    sysroot_repo(
        name = "{}_build".format(prefix),
        libc_version = libc_version,
        supported_archs = supported_archs,
        variant = "build",
        packages = _DEBIAN12_RUNTIME_PKGS + _DEBIAN12_BUILD_PKGS + extra_runtime_packages + extra_build_packages,
        extra_compile_flags = extra_build_compile_flags,
        extra_link_flags = extra_build_link_flags,
        target_settings = build_target_settings,
    )

    sysroot_repo(
        name = "{}_test".format(prefix),
        libc_version = libc_version,
        supported_archs = supported_archs,
        variant = "test",
        packages = _DEBIAN12_RUNTIME_PKGS + _DEBIAN12_BUILD_PKGS + _DEBIAN12_TEST_PKGS + extra_runtime_packages + extra_build_packages + extra_test_packages,

        # TODO(james) figure out if we need the build compile flags here or not. The original CUDA implementation
        # seemed to require it.
        extra_compile_flags = extra_build_compile_flags,
        extra_link_flags = extra_build_link_flags,
        target_settings = test_target_settings,
    )

def _debian12_sysroots():
    _debian12_with_extras(
        prefix = "sysroot_debian12",
        runtime_target_settings = ["@gml//bazel/cc_toolchains/sysroots:use_debian12_runtime_sysroot"],
        build_target_settings = ["@gml//bazel/cc_toolchains/sysroots:use_debian12_build_sysroot"],
        test_target_settings = ["@gml//bazel/cc_toolchains/sysroots:use_debian12_test_sysroot"],
    )

def _jetson_sysroots():
    sysroot_type_setting = "@gml//bazel/cc_toolchains/sysroots:sysroot_type_jetson"
    runtime_pkgs = [
        "ubuntu2004_ca-certificates",
        "ubuntu2004_libtinfo6",
        "ubuntu2004_libc6",
        "ubuntu2004_libelf1",
        "ubuntu2004_libstdc++6",
        "ubuntu2004_zlib1g",
        "ubuntu2004_libunwind8",
        # NVIDIA's container runtime requires that `ldconfig` exist in the container.
        "ubuntu2004_libc-bin",
    ]
    sysroot_repo(
        name = "sysroot_jetson_runtime",
        libc_version = "glibc2_31",
        supported_archs = ["aarch64"],
        variant = "runtime",
        packages = runtime_pkgs,
        target_settings = [sysroot_type_setting],
    )
    build_pkgs = [
        "ubuntu2004_liblzma-dev",
        "ubuntu2004_libunwind-dev",
        "ubuntu2004_libncurses-dev",
        "ubuntu2004_libc6-dev",
        "ubuntu2004_libegl1-mesa-dev",
        "ubuntu2004_libelf-dev",
        "ubuntu2004_libgcc-9-dev",
        "ubuntu2004_libgles2-mesa-dev",
        "ubuntu2004_libicu-dev",
        "ubuntu2004_libstdc++-9-dev",
        "ubuntu2004_linux-libc-dev",
        "ubuntu2004_mesa-common-dev",
        "ubuntu2004_zlib1g-dev",
        "ubuntu2004_libva-dev",
        "jetson_libnvinfer-dev",
        "jetson_libnvinfer-plugin8",
        "jetson_libnvonnxparsers-dev",
        "jetson_cuda-nvcc-11-4",
        "jetson_nvidia-l4t-camera",
        "jetson_nvidia-l4t-core",
        "jetson_nvidia-l4t-cuda",
        "jetson_nvidia-l4t-jetson-multimedia-api",
        "jetson_nvidia-l4t-kernel-headers",
        "jetson_nvidia-l4t-multimedia",
        "jetson_nvidia-l4t-multimedia-utils",
    ]
    sysroot_repo(
        name = "sysroot_jetson_build",
        libc_version = "glibc2_31",
        supported_archs = ["aarch64"],
        variant = "build",
        packages = runtime_pkgs + build_pkgs,
        target_settings = [sysroot_type_setting],
        extra_compile_flags = [
            # Order is important here. Bazel doesn't like the symlink from jetson_multimedia_api/include to ../argus/include,
            # so we need to make sure the compiler resolves the actual argus includes first.
            "-isystem%sysroot%/usr/src/jetson_multimedia_api/argus/include",
            "-isystem%sysroot%/usr/src/jetson_multimedia_api/include",
            "-isystem%sysroot%/usr/local/cuda-11.4/targets/aarch64-linux/include",
        ],
        extra_link_flags = [
            "-L%sysroot%/usr/lib/aarch64-linux-gnu/tegra",
            "-L%sysroot%/usr/local/cuda-11.4/targets/aarch64-linux/lib",
        ],
        path_prefix_filters = _DEFAULT_BUILD_PATH_PREFIXES + ["usr/src/jetson_multimedia_api"],
    )
    test_pkgs = [
        "ubuntu2004_bash",
        "ubuntu2004_grep",
        "ubuntu2004_sed",
    ]
    sysroot_repo(
        name = "sysroot_jetson_test",
        libc_version = "glibc2_31",
        supported_archs = ["aarch64"],
        variant = "test",
        packages = runtime_pkgs + build_pkgs + test_pkgs,
        target_settings = [sysroot_type_setting],
    )

def _intel_gpu_sysroots():
    sysroot_repo(
        name = "sysroot_intelgpu_runtime",
        libc_version = "glibc2_36",
        supported_archs = ["x86_64"],
        variant = "runtime",
        packages = _DEBIAN12_RUNTIME_PKGS + [
            "intel-compute-runtime_level-zero-gpu",
            "intel-compute-runtime_opencl-icd",
            "intel-compute-runtime_igc-core",
            "intel-compute-runtime_igc-opencl",
            "intel-compute-runtime_libigdgmm",
        ],
        target_settings = ["@gml//bazel/cc_toolchains/sysroots:sysroot_type_intelgpu"],
    )

PYTHON_RUNTIME_PKGS = [
    "debian12_coreutils",
    "debian12_python3.11",
]

def _cuda_sysroot():
    ffmpeg_runtime_pkgs = [
        "debian12_libvdpau1",
        "debian12_libvpx-dev",
        "nvidia_libnpp-12-3",
    ]

    ffmpeg_build_pkgs = [
        "debian12_nasm",
        "debian12_yasm",
        "nvidia_cuda-toolkit-12-3",
    ]

    target_settings = ["@gml//bazel/cc_toolchains/sysroots:sysroot_type_cuda"]
    _debian12_with_extras(
        prefix = "sysroot_cuda",
        runtime_target_settings = target_settings,
        build_target_settings = target_settings,
        test_target_settings = target_settings,
        extra_runtime_packages = PYTHON_RUNTIME_PKGS + ffmpeg_runtime_pkgs,
        extra_build_packages = ffmpeg_build_pkgs,
        supported_archs = ["x86_64"],
        extra_build_compile_flags = [
            "-isystem%sysroot%/usr/local/cuda/include",
        ],
        extra_build_link_flags = [
            "-L%sysroot%/usr/local/cuda/lib64",
        ],
    )

def _experimental_sysroot():
    target_settings = ["@gml//bazel/cc_toolchains/sysroots:sysroot_type_experimental"]
    _debian12_with_extras(
        prefix = "sysroot_experimental",
        runtime_target_settings = target_settings,
        build_target_settings = target_settings,
        test_target_settings = target_settings,
        extra_runtime_packages = PYTHON_RUNTIME_PKGS + ["debian12_libmagic1", "debian12_dash"],
    )

def _gml_sysroots():
    _debian12_sysroots()
    _jetson_sysroots()
    _intel_gpu_sysroots()
    _cuda_sysroot()
    _experimental_sysroot()

SYSROOT_LIBC_VERSIONS = [
    "glibc2_36",
    "glibc2_31",
]

SYSROOT_ARCHITECTURES = [
    "aarch64",
    "x86_64",
]

gml_sysroots = _gml_sysroots
