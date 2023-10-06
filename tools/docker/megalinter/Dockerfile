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

FROM docker.io/oxsecurity/megalinter-cupcake:v7@sha256:171f2f5049a598aee1c752ebe830dc17c81ae5a003aabc08b87f09edada5f78f

COPY .tflint.hcl $HOME/.tflint.hcl

RUN apk add clang-extra-tools && \
    go install github.com/bazelbuild/buildtools/buildifier@v0.0.0-20230926111657-7d855c59baeb  && \
    go install github.com/bufbuild/buf/cmd/buf@v1.26.1 && \
    go clean -modcache && \
    go clean -cache && \
    npm install -g squawk-cli@0.24.0 && \
    npm cache clean --force && \
    tflint --init
