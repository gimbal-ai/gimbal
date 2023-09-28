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

## Bazel command to use.
BAZEL      := bazelisk
## Lint runner
MEGALINTER := mega-linter-runner

# This default target is invoked by CodeQL. Ensure that this is the first
# target in this makefile.
.PHONY: go
go: ## A simple go build that ensure that the go code compiles.
	CGO_ENABLED=0 go build ./...

.PHONY: go-mod-tidy
go-mod-tidy: ## Ensure that go are cleaned up.
	go mod tidy -compat=1.21

.PHONY: gazelle
gazelle: update-python-manifest ## Run gazelle.
	$(BAZEL) run //:gazelle

.PHONY: go-setup
go-setup: go-mod-tidy gazelle ## Run go setup to regenrate modules/build files.

.PHONY: update-python-manifest
update-python-manifest:
	$(BAZEL) run //:gazelle_python_manifest.update

.PHONY: lint
lint:
	$(MEGALINTER) \
		--fix \
		--env REPORT_OUTPUT_FOLDER=none \
		--image=us-docker.pkg.dev/gimlet-dev-infra-0/gimlet-dev-infra-public-docker-artifacts/megalinter-gml-custom:20230927134618
