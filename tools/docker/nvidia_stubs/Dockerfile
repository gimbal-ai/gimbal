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

FROM nvidia/cuda:11.4.3-runtime-ubuntu20.04

RUN apt-get update && \
  apt-get install -y \
  build-essential \
  libnvinfer8=8.5.2-1+cuda11.8 \
  libnvinfer-plugin8=8.5.2-1+cuda11.8 \
  libnvonnxparsers8=8.5.2-1+cuda11.8

COPY stubify.sh scripts/stubify.sh
COPY stubify_all.sh /scripts/stubify_all.sh

ENTRYPOINT ["/scripts/stubify_all.sh"]
