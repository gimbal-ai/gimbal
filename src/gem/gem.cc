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
#include "src/gem/controller/controller.h"
#include "src/gem/controller/lifecycle.h"
#include "src/shared/version/version.h"

using ::gml::gem::controller::Controller;
using ::gml::gem::controller::DefaultDeathHandler;
using ::gml::gem::controller::TerminationHandler;

DEFINE_string(deploy_key, gflags::StringFromEnv("GML_DEPLOY_KEY", ""),
              "The deploy key used to connect to the control plane");
DEFINE_string(controlplane_addr,
              gflags::StringFromEnv("GML_CONTROLPLANE_ADDR", "app.gimletlabs.ai:443"),
              "The address of the controlplane");

int main(int argc, char** argv) {
  gml::EnvironmentGuard env_guard(&argc, argv);
  DefaultDeathHandler err_handler;
  // This covers signals such as SIGSEGV and other fatal errors.
  // We print the stack trace and die.
  auto signal_action = std::make_unique<gml::SignalAction>();
  signal_action->RegisterFatalErrorHandler(err_handler);

  // Install signal handlers where graceful exit is possible.
  TerminationHandler::InstallSignalHandlers();

  LOG(INFO) << "Starting GEM";

  auto controller = std::make_unique<Controller>(FLAGS_deploy_key, FLAGS_controlplane_addr);
  TerminationHandler::set_controller(controller.get());

  GML_CHECK_OK(controller->Init());
  GML_CHECK_OK(controller->Run());
  GML_CHECK_OK(controller->Stop(std::chrono::seconds(1)));

  TerminationHandler::set_controller(controller.get());

  return 0;
}
