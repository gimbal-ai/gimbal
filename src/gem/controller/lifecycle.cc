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

#include "src/gem/controller/lifecycle.h"

#include "src/gem/controller/controller.h"

namespace gml::gem::controller {

void TerminationHandler::OnTerminate(int signum) {
  if (controller_ != nullptr) {
    LOG(INFO) << "Trying to gracefully stop agent manager";
    auto s = controller_->Stop(std::chrono::seconds{5});
    if (!s.ok()) {
      LOG(ERROR) << "Failed to gracefully stop agent manager, it will terminate shortly.";
    }
    exit(signum);
  }
}

}  // namespace gml::gem::controller
