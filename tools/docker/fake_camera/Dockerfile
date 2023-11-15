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

FROM ubuntu:22.04@sha256:2b7412e6465c3c7fc5bb21d3e6f1917c167358449fecac8176c6e496e5c1f05f
ENV GCSFUSE_REPO gcsfuse-jammy

RUN apt-get update && apt-get install --yes --no-install-recommends \
  ca-certificates curl gnupg ffmpeg v4l-utils \
  && echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" \
  | tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update && apt-get install --yes gcsfuse \
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN mkdir /videos

WORKDIR /app
ADD loop_video.sh ./

ENV VIDEO_DEVICE=/dev/video0
ENV GCS_BUCKET=gml-dev-infra-videos
ENV VIDEO_FILE=gimlets/tictoc.mp4

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["/app/loop_video.sh"]
