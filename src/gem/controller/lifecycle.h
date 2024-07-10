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

#pragma once

#include "src/common/base/base.h"
#include "src/common/base/logging.h"
#include "src/common/signal/signal.h"

namespace gml::gem::controller {

class ControllerBase;

class DefaultDeathHandler : public gml::FatalErrorHandlerInterface {
 public:
  DefaultDeathHandler() = default;
  void OnFatalError() const override {
    // Stack trace will print automatically; any additional state dumps can be done here.
    // Note that actions here must be async-signal-safe and must not allocate memory.
  }
};

// Signal handlers for graceful termination.
class TerminationHandler {
 public:
  // This list covers signals that are handled gracefully.
  static constexpr auto kSignals = ::gml::MakeArray(SIGINT, SIGQUIT, SIGTERM, SIGHUP);

  static void InstallSignalHandlers() {
    for (int kSignal : kSignals) {
      signal(kSignal, TerminationHandler::OnTerminate);
    }
  }

  static void set_controller(ControllerBase* controller) { controller_ = controller; }

  static void OnTerminate(int signum);

 private:
  inline static ControllerBase* controller_ = nullptr;
};

}  // namespace gml::gem::controller
