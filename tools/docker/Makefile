# Copyright © 2023- Gimlet Labs Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Gimlet Labs Inc. and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Gimlet Labs Inc. and its suppliers and
# may be covered by U.S. and Foreign Patents, patents in process,
# and are protected by trade secret or copyright law. Dissemination
# of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from
# Gimlet Labs Inc.
#
# SPDX-License-Identifier: Proprietary

# Commands
DOCKER := docker
BUILD_DIR ?= .build

# We use our dev image to build some other tools. This fetches the latest tag.
TOT		       := $$(git rev-parse --show-toplevel)
DOCKER_PROPERTIES_FILE := $(TOT)/docker.properties

## Clang deb parameters
CLANG_VERSION := 15.0
CLANG_SUFFIX := $(shell date +%Y%m%d%H%M%S)

CLANG_TAG="$(CLANG_VERSION)-$(CLANG_SUFFIX)"

clang_deb_fname := "clang-$(CLANG_TAG).deb"
clang_linters_deb_fname := "clang-linters-$(CLANG_TAG).deb"
clang_gs_path :=  gs://gimlet-dev-infra/clang/$(CLANG_TAG)
clang_deb_image_tag := "gcr.io/gimlet-dev-infra-0/gimlet-dev-infra/clang_deb_builder_image:$(CLANG_VERSION)"
CLANG_BUILD_DIR := "$(BUILD_DIR)/clang-$(CLANG_TAG)"

## Sysroot parameters
SYSROOT_REV := $(shell date +%Y%m%d%H%M%S)
SYSROOT_BUILD_DIR := $(BUILD_DIR)/sysroots
SYSROOT_ARCHITECTURES := amd64 arm64
SYSROOT_GS_PATH := gs://gimlet-dev-infra/sysroots/$(SYSROOT_REV)
SYSROOT_CREATOR_IMAGE_TAG := sysroot-creator-$(SYSROOT_REV)

##############################################
# Clang Build
##############################################
.PHONY: build_clang_deb_image
build_clang_deb_image:
	$(DOCKER) build clang_deb_image -t $(clang_deb_image_tag)

.PHONY: upload_clang_deb
upload_clang_deb: build_clang_deb_image ## Target to build and upload clang deb image
	@mkdir -p $(CLANG_BUILD_DIR)
	$(DOCKER) run --rm -e CLANG_SUFFIX=$(CLANG_SUFFIX) -e CLANG_VERSION=$(CLANG_VERSION) -v $(PWD)/$(CLANG_BUILD_DIR):/image $(clang_deb_image_tag)

	sha256sum $(CLANG_BUILD_DIR)/* > $(CLANG_BUILD_DIR)/sha256sums
	gsutil cp $(CLANG_BUILD_DIR)/* $(clang_gs_path)/

	cat $(CLANG_BUILD_DIR)/sha256sums


##############################################
# Sysroots Build
##############################################
.PHONY: build_all_sysroots
build_all_sysroots: sysroot_creator/Dockerfile sysroot_creator/build_all_sysroots.sh
	@mkdir -p $(SYSROOT_BUILD_DIR)
	$(DOCKER) build sysroot_creator -t $(SYSROOT_CREATOR_IMAGE_TAG)
	./sysroot_creator/build_all_sysroots.sh $(SYSROOT_BUILD_DIR) $(SYSROOT_CREATOR_IMAGE_TAG)

.PHONY: upload_all_sysroots
upload_all_sysroots: build_all_sysroots
	gsutil cp $(SYSROOT_BUILD_DIR)/* $(SYSROOT_GS_PATH)
	sha256sum $(SYSROOT_BUILD_DIR)/*
