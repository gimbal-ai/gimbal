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
