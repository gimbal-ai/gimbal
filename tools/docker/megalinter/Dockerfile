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

FROM docker.io/oxsecurity/megalinter-cupcake:v7@sha256:6c7e48d9832345f3bd06750a5ba35e8ad1e1d887b0603b414c022e8e0c5d4154

ARG GITHUB_TOKEN

COPY .tflint.hcl /.tflint.hcl

RUN echo "@edge https://dl-cdn.alpinelinux.org/alpine/edge/community" >> /etc/apk/repositories && \
    apk add go@edge

RUN apk add clang-extra-tools g++ && \
    go install github.com/bazelbuild/buildtools/buildifier@v0.0.0-20240422193413-1429e15ae755 && \
    go install github.com/bufbuild/buf/cmd/buf@v1.26.1 && \
    go clean -modcache && \
    go clean -cache && \
    npm install -g squawk-cli@0.24.0 && \
    npm cache clean --force && \
    tflint --init
