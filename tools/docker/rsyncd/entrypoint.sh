#!/bin/sh

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

mkdir -p "${DATA_PATH}"

cat <<EOF >/etc/rsyncd.conf
uid = ${USER}
gid = ${GROUP}
use chroot = yes
log file = /dev/stdout
reverse lookup = no
[data]
  hosts allow = *
  read only = false
  path = ${DATA_PATH}
EOF

exec /usr/bin/rsync --no-detach --daemon --config /etc/rsyncd.conf
