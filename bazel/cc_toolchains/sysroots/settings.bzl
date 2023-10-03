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

# THIS FILE IS GENERATED BY build_all_sysroots

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

def _sysroot_settings():
    bool_flag(
        name = "sysroot_enable_debug",
        build_setting_default = False,
    )
    native.config_setting(
        name = "sysroot_debug_enabled",
        flag_values = {
            ":sysroot_enable_debug": "True",
        },
    )
    native.config_setting(
        name = "sysroot_debug_disabled",
        flag_values = {
            ":sysroot_enable_debug": "False",
        },
    )

sysroot_settings = _sysroot_settings
