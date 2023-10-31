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

def _collect_shared_libs(foreign_cc_target, libs):
    for lib in libs:
        native.filegroup(
            name = lib.partition(".")[0],
            srcs = [foreign_cc_target],
            output_group = lib,
            visibility = ["//visibility:public"],
        )

    native.filegroup(
        name = "shared_libs",
        srcs = [":" + lib.partition(".")[0] for lib in libs],
        visibility = ["//visibility:public"],
    )

collect_shared_libs = _collect_shared_libs