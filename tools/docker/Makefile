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

## Clang deb parameters
CLANG_VERSION := 15.0
CLANG_SUFFIX := $(shell date +%Y%m%d%H%M%S)

CLANG_TAG="$(CLANG_VERSION)-$(CLANG_SUFFIX)"

clang_deb_fname := "clang-$(CLANG_TAG).deb"
clang_linters_deb_fname := "clang-linters-$(CLANG_TAG).deb"
clang_gs_path :=  gs://gimlet-dev-infra-public/clang/$(CLANG_TAG)
clang_deb_image_tag := "clang_deb_builder_image:$(CLANG_VERSION)"
CLANG_BUILD_DIR := "$(BUILD_DIR)/clang-$(CLANG_TAG)"

## Megalinter custom image
REPOSITORY := us-docker.pkg.dev/gimlet-dev-infra-0/gimlet-dev-infra-public-docker-artifacts
MEGALINTER_BUILD_DIR := $(BUILD_DIR)/megalinter
MEGALINTER_REV := $(shell date +%Y%m%d%H%M%S)
MEGALINTER_IMAGE_TAG := "$(REPOSITORY)/megalinter-gml-custom:$(MEGALINTER_REV)"

## NVIDIA stub creator
NVIDIA_STUB_REV := $(shell date +%Y%m%d%H%M%S)
NVIDIA_STUB_BUILD_DIR := $(BUILD_DIR)/nvidia_stubs
NVIDIA_STUB_CREATOR_IMAGE_TAG := nvidia-stub-creator
NVIDIA_STUB_GS_PATH := gs://gimlet-dev-infra-public/nvidia-stubs/$(NVIDIA_STUB_REV)
NVIDIA_STUB_TARGETS := $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs_amd64.tar.gz $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs_arm64.tar.gz

## rsyncd image
RSYNCD_REV := $(shell date +%Y%m%d%H%M%S)
RSYNCD_IMAGE_TAG := "$(REPOSITORY)/rsyncd-image:$(RSYNCD_REV)"

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
# Megalinter Build
##############################################
megalinter/.tflint.hcl: ../../.tflint.hcl
	@cp ../../.tflint.hcl $@
.PHONY: build_and_push_megalinter
build_and_push_megalinter: megalinter/Dockerfile megalinter/.tflint.hcl
	@mkdir -p $(MEGALINTER_BUILD_DIR)
	DOCKER_BUILDKIT=1 $(DOCKER) build megalinter -t $(MEGALINTER_IMAGE_TAG) --push
	$(MAKE) -C ../chef sync_images

##############################################
# NVIDIA Stub libraries
##############################################
$(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs_%.tar: nvidia_stubs/Dockerfile nvidia_stubs/stubify.sh nvidia_stubs/stubify_all.sh
	@mkdir -p $(NVIDIA_STUB_BUILD_DIR)
	DOCKER_BUILDKIT=1 $(DOCKER) build --platform=$* nvidia_stubs -t $(NVIDIA_STUB_CREATOR_IMAGE_TAG):$*
	$(DOCKER) run -it -v $(PWD)/$(NVIDIA_STUB_BUILD_DIR):/build $(NVIDIA_STUB_CREATOR_IMAGE_TAG):$* /build/nvidia_stubs_$*.tar $*

$(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs.tar.gz: $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs_amd64.tar $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs_arm64.tar
	@cp $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs_amd64.tar $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs.tar
	@tar --concatenate -f $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs.tar $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs_arm64.tar
	@gzip -f $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs.tar

.PHONY: upload_nvidia_stubs
upload_nvidia_stubs: $(NVIDIA_STUB_BUILD_DIR)/nvidia_stubs.tar.gz
	gsutil cp $< $(NVIDIA_STUB_GS_PATH)/nvidia_stubs.tar.gz
	sha256sum $<

##############################################
# rsyncd image
##############################################
.PHONY: build_and_push_rsyncd
build_and_push_rsyncd: rsyncd/Dockerfile rsyncd/entrypoint.sh
	DOCKER_BUILDKIT=1 $(DOCKER) build rsyncd -t $(RSYNCD_IMAGE_TAG) --push
