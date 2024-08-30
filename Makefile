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
	go mod tidy -compat=1.22

.PHONY: gazelle
gazelle: go-mod-tidy update-python-manifest ## Run gazelle.
	$(BAZEL) run //:gazelle

requirements_lock.txt: requirements.in constraints.txt
	$(BAZEL) run //:requirements.update

src/experimental/requirements_lock.txt: src/experimental/requirements.in src/experimental/constraints.txt
	$(BAZEL) run //src/experimental:requirements.update

src/api/python/requirements_lock.txt: src/api/python/requirements.in
	$(BAZEL) run //src/api/python:requirements.update

.PHONY: update-python-requirements
update-python-requirements: requirements_lock.txt src/experimental/requirements_lock.txt src/api/python/requirements_lock.txt
	$(BAZEL) mod deps --lockfile_mode=update

gazelle_python.yaml: requirements_lock.txt
	$(BAZEL) run //:gazelle_python_manifest.update

src/experimental/gazelle_python.yaml: src/experimental/requirements_lock.txt
	$(BAZEL) run //src/experimental:gazelle_python_manifest.update

src/api/python/gazelle_python.yaml: src/api/python/requirements_lock.txt
	$(BAZEL) run //src/api/python:gazelle_python_manifest.update

.PHONY: update-python-manifest
update-python-manifest: gazelle_python.yaml src/experimental/gazelle_python.yaml src/api/python/gazelle_python.yaml

.PHONY: update-python-requirements-clean
update-python-requirements-clean:
	rm requirements_lock.txt
	touch requirements_lock.txt
	$(BAZEL) run //:requirements.update
	rm src/experimental/requirements_lock.txt
	touch src/experimental/requirements_lock.txt
	$(BAZEL) run //src/experimental:requirements.update
	rm src/api/python/requirements_lock.txt
	touch src/api/python/requirements_lock.txt
	$(BAZEL) run //src/api/python:requirements.update

src/ui/node_modules: src/ui/package.json src/ui/pnpm-lock.yaml
	cd src/ui && pnpm install

.PHONY: jest
jest: src/ui/node_modules
	cd src/ui && pnpm jest

.PHONY: jestu
jestu: src/ui/node_modules
	cd src/ui && pnpm jest -u

.PHONY: devui
devui: src/ui/node_modules
	cd src/ui && pnpm dev

.PHONY: storybook
storybook: src/ui/node_modules
	cd src/ui && pnpm storybook-dev

.PHONY: genfiles
genfiles:
	scripts/update_graphql_types.sh
	scripts/update_gen_protos.sh
	scripts/update_visspec_types.sh
	scripts/update_globalcss.sh
	scripts/update_ui_srcs.sh
	go generate ./...

.PHONY: lint
lint: src/ui/node_modules
	$(MEGALINTER) \
		--fix \
		--nodockerpull \
		--env VALIDATE_ALL_CODEBASE=false \
		--env NODE_PATH=./src/ui/node_modules \
		--env REPORT_OUTPUT_FOLDER=none \
		--env GITHUB_TOKEN="${GITHUB_TOKEN}" \
		--image=us-docker.pkg.dev/gimlet-dev-infra-0/gimlet-dev-infra-public-docker-artifacts/megalinter-gml-custom:20240709133108

.PHONY: lint-all
lint-all: src/ui/node_modules
	$(MEGALINTER) \
		--nodockerpull \
		--env APPLY_FIXES=none \
		--env VALIDATE_ALL_CODEBASE=true \
		--env NODE_PATH=./src/ui/node_modules \
		--env REPORT_OUTPUT_FOLDER=none \
		--env GITHUB_TOKEN="${GITHUB_TOKEN}" \
		--image=us-docker.pkg.dev/gimlet-dev-infra-0/gimlet-dev-infra-public-docker-artifacts/megalinter-gml-custom:20240709133108
