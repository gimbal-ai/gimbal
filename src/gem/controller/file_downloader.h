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

#include <filesystem>
#include <string>

#include <absl/container/flat_hash_map.h>
#include <sole.hpp>

#include "src/gem/controller/message_handler.h"

namespace gml::gem::controller {

// Forward declare the downloader task.
class FileDownloaderTask;

class FileDownloader : public MessageHandler {
 public:
  FileDownloader() = delete;
  FileDownloader(gml::event::Dispatcher*, GEMInfo*, GRPCBridge*);

  Status Init() override;
  Status BlockingDownload(const sole::uuid& fid, const std::string& sha256sum, size_t size,
                          const std::filesystem::path& path);
  Status HandleMessage(const internal::controlplane::egw::v1::BridgeResponse& msg) override;
  Status Finish() override;

 private:
  absl::Mutex downloader_mu_;
  absl::flat_hash_map<sole::uuid, FileDownloaderTask*> downloaders_ ABSL_GUARDED_BY(downloader_mu_);
  absl::flat_hash_map<sole::uuid, event::RunnableAsyncTaskUPtr> downloader_tasks_
      ABSL_GUARDED_BY(downloader_mu_);
  //  sole::uuid fid_;
  //  int size_;
  std::string sha256sum_;
};

}  // namespace gml::gem::controller
