/*
 * Copyright 2023- Gimlet Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "src/gem/controller/lifecycle.h"
#include "src/gem/fakegem/controller.h"

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

  LOG(INFO) << "Starting Replay GEM";

  auto controller = std::make_unique<gml::gem::fakegem::FakeController>(FLAGS_deploy_key,
                                                                        FLAGS_controlplane_addr);
  TerminationHandler::set_controller(controller.get());

  GML_CHECK_OK(controller->Init());
  GML_CHECK_OK(controller->Run());
  GML_CHECK_OK(controller->Stop(std::chrono::seconds(1)));

  TerminationHandler::set_controller(controller.get());

  return 0;
}
