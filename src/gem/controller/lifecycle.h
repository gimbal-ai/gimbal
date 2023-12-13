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
