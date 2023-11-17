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

OUT="$(basename "$VIDEO_FILE")"
gsutil cp "$GCS_BUCKET$VIDEO_FILE" "$OUT"

if [[ $AUTOSELECT_DEVICE == "true" ]]; then
  VIDEO_DEVICE=$(find /dev/video* | tail -1)
fi

ffmpeg -stream_loop -1 -re -i "$OUT" -vf format=yuv420p -f v4l2 "$VIDEO_DEVICE"
