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
