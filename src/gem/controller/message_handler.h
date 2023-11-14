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
