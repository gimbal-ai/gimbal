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


## Sysroot parameters
SYSROOT_REV := $(shell date +%Y%m%d%H%M%S)
SYSROOT_BUILD_DIR := $(BUILD_DIR)/sysroots
SYSROOT_ARCHITECTURES := amd64 arm64
SYSROOT_GS_PATH := gs://gimlet-dev-infra/sysroots/$(SYSROOT_REV)
SYSROOT_CREATOR_IMAGE_TAG := sysroot-creator-$(SYSROOT_REV)

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
