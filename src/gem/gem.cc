/*
 * Copyright © 2023- Gimlet Labs, Inc.
 * All Rights Reserved.
 *
 * NOTICE:  All information contained herein is, and remains
 * the property of Gimlet Labs, Inc. and its suppliers,
 * if any.  The intellectual and technical concepts contained
 * herein are proprietary to Gimlet Labs, Inc. and its suppliers and
 * may be covered by U.S. and Foreign Patents, patents in process,
 * and are protected by trade secret or copyright law. Dissemination
 * of this information or reproduction of this material is strictly
 * forbidden unless prior written permission is obtained from
 * Gimlet Labs, Inc.
 *
 * SPDX-License-Identifier: Proprietary
 */

#include "src/common/base/base.h"
#include "src/shared/version/version.h"

DEFINE_string(deploy_key, gflags::StringFromEnv("GML_DEPLOY_KEY", ""),
              "The deploy key used to connect to the control plane");
DEFINE_string(controlplane_addr,
              gflags::StringFromEnv("GML_CONTROLPLANE_ADDR", "app.gimletlabs.ai:443"),
              "The address of the controlplane");

int main(int argc, char** argv) {
  gml::EnvironmentGuard env_guard(&argc, argv);
  LOG(INFO) << "Starting GEM";

  return 0;
}
