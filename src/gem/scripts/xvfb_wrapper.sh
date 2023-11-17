#!/bin/bash -e

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

# Stop xvfb-run complaining about awk missing.
# It doesn't use awk for anything useful, but this stops the error log.
# TODO(james): sysroot_creator should support update-alternatives which would remove the need for this.
ln -sf /usr/bin/gawk /usr/bin/awk

xvfb-run -a "$@"