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

#include "src/common/event/event.h"
#include "src/gem/controller/gem_info.h"
#include "src/gem/controller/grpc_bridge.h"

namespace gml::gem::controller {

class MessageHandler {
 public:
  // Force initialization by subclasses.
  MessageHandler() = delete;

  /**
   * MessageHandler handles messages asynchronously and may respond over the paseed on GRPC bridge
   * connection.
   */
  MessageHandler(event::Dispatcher* dispatcher, GEMInfo* agent_info, GRPCBridge* bridge)
      : dispatcher_(dispatcher), agent_info_(agent_info), bridge_(bridge){};

  virtual ~MessageHandler() = default;

  /**
   * Init is called before any messages are sent.
   */
  virtual Status Init() = 0;

  /**
   * Handle a message of the registered type. This function is called using the event loop thread.
   * Do not call blocking operators while handling the message.
   */
  virtual Status HandleMessage(const gml::internal::controlplane::egw::v1::BridgeResponse& msg) = 0;

  /**
   * Is called when no more messages will be sent.
   *
   * This gives a chance for cleanup before the destructor is called.
   */
  virtual Status Finish() = 0;

 protected:
  const GEMInfo* agent_info() const { return agent_info_; }
  GRPCBridge* bridge() const { return bridge_; }
  gml::event::Dispatcher* dispatcher() { return dispatcher_; }

 private:
  gml::event::Dispatcher* dispatcher_ = nullptr;
  const GEMInfo* agent_info_;
  GRPCBridge* bridge_;
};

}  // namespace gml::gem::controller
