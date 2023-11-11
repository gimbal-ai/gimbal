# Copyright © 2023- Gimlet Labs, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Gimlet Labs, Inc. and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Gimlet Labs, Inc. and its suppliers and
# may be covered by U.S. and Foreign Patents, patents in process,
# and are protected by trade secret or copyright law. Dissemination
# of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from
# Gimlet Labs, Inc.
#
# SPDX-License-Identifier: Proprietary

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:gml_build_system.bzl", "gml_cc_library")

PLUGINS = dict(
    argus = struct(
        enabled_by_default = False,
        libs = [
            "//src/gem/calculators/plugin/argus:cc_library",
        ],
    ),
    cpu_tensor = struct(
        enabled_by_default = True,
        libs = [
            "//src/gem/calculators/plugin/cpu_tensor:cc_library",
            "//src/gem/exec/plugin/cpu_tensor:cc_library",
            "//src/gem/build/plugin/cpu_tensor:cc_library",
        ],
    ),
    ffmpeg = struct(
        enabled_by_default = True,
        libs = [
            "//src/gem/calculators/plugin/ffmpeg:cc_library",
        ],
    ),
    tensorrt = struct(
        enabled_by_default = False,
        libs = [
            "//src/gem/calculators/plugin/tensorrt:cc_library",
            "//src/gem/exec/plugin/tensorrt:cc_library",
            "//src/gem/build/plugin/tensorrt:cc_library",
        ],
    ),
    opencv_cam = struct(
        enabled_by_default = True,
        libs = [
            "//src/gem/calculators/plugin/opencv_cam:cc_library",
        ],
    ),
)

def _plugin_enabled(plugin_name):
    return "//src/gem/plugins:{}_enabled".format(plugin_name)

def _select_plugins():
    selects = []
    for name, plugin in PLUGINS.items():
        selects += select({
            _plugin_enabled(name): plugin.libs,
            "//conditions:default": [],
        })
    return selects

def _plugin_settings():
    for name, plugin in PLUGINS.items():
        bool_flag(
            name = "enable_{}".format(name),
            build_setting_default = plugin.enabled_by_default,
        )
        native.config_setting(
            name = "{}_enabled".format(name),
            flag_values = {
                ":enable_{}".format(name): "True",
            },
        )
        native.config_setting(
            name = "{}_disabled".format(name),
            flag_values = {
                ":enable_{}".format(name): "False",
            },
        )

def gem_plugin_cc_library(name, plugin_name, **kwargs):
    if plugin_name not in PLUGINS:
        fail("{} not found in PLUGINS. Add new plugins to src/gem/plugins/plugins.bzl:PLUGINS".format(plugin_name))

    target_compatible_with = kwargs.pop("target_compatible_with", [])

    target_compatible_with = target_compatible_with + select({
        _plugin_enabled(plugin_name): [],
        "//conditions:default": ["@platforms//:incompatible"],
    })

    gml_cc_library(
        name = name,
        target_compatible_with = target_compatible_with,
        **kwargs
    )

plugin_settings = _plugin_settings
select_plugins = _select_plugins