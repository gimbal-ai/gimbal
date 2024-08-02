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

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")
load("//bazel:gml_build_system.bzl", "gml_cc_library")

PLUGINS = dict(
    argus = struct(
        enabled_by_default = False,
        libs = [
            "//src/gem/calculators/plugin/argus:cc_library",
            "//src/gem/capabilities/plugin/argus:cc_library",
            "//src/gem/calculators/plugin/argus/optionspb:optionspb_options_registry",
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
    huggingface = struct(
        enabled_by_default = True,
        libs = [
            "//src/gem/exec/plugin/huggingface:cc_library",
            "//src/gem/calculators/plugin/huggingface:cc_library",
        ],
    ),
    tensorrt = struct(
        enabled_by_default = False,
        libs = [
            "//src/gem/calculators/plugin/tensorrt:cc_library",
            "//src/gem/exec/plugin/tensorrt:cc_library",
            "//src/gem/build/plugin/tensorrt:cc_library",
            "//src/gem/capabilities/plugin/tensorrt:cc_library",
        ],
    ),
    opencv_cam = struct(
        enabled_by_default = True,
        libs = [
            "//src/gem/calculators/plugin/opencv_cam:cc_library",
            "//src/gem/capabilities/plugin/opencv_cam:cc_library",
            "//src/gem/calculators/plugin/opencv_cam/optionspb:optionspb_options_registry",
        ],
    ),
    openvino = struct(
        enabled_by_default = True,
        libs = [
            "//src/gem/exec/plugin/openvino:cc_library",
            "//src/gem/build/plugin/openvino:cc_library",
            "//src/gem/calculators/plugin/openvino:cc_library",
            "//src/gem/capabilities/plugin/openvino:cc_library",
            "//src/gem/metrics/plugin/openvino:cc_library",
        ],
    ),
    intelgpu = struct(
        enabled_by_default = False,
        libs = [
            "//src/gem/metrics/plugin/intelgpu:cc_library",
        ],
    ),
    jetson = struct(
        enabled_by_default = False,
        libs = [
            "//src/gem/metrics/plugin/jetson:cc_library",
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
