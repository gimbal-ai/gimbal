# Copyright 2018- The Pixie Authors.
# Modifications Copyright 2023- Gimlet Labs, Inc.
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

FROM debian:bookworm@sha256:fab22df37377621693c68650b06680c0d8f7c6bf816ec92637944778db3ca2c0

RUN apt-get update -y && \
  apt-get install -y golang curl xz-utils

RUN curl -L https://github.com/mikefarah/yq/releases/download/v4.30.8/yq_linux_amd64 -o /bin/yq && \
  chmod +x /bin/yq

VOLUME /pkgdb

COPY package_satisifier /package_satisifier
WORKDIR /package_satisifier
RUN go mod init "package_satisifier" && \
  go mod tidy && \
  go build .

WORKDIR /
COPY build_tar_for_packages.sh /scripts/build_tar_for_packages.sh
COPY features /feature_yamls

ENTRYPOINT ["/scripts/build_tar_for_packages.sh", "/package_satisifier/package_satisifier", "/feature_yamls"]

VOLUME /build
