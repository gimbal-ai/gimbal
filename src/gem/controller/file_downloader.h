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

#include <filesystem>
#include <future>
#include <memory>
#include <string>

#include <absl/container/flat_hash_map.h>
#include <sole.hpp>

#include "src/common/event/task.h"
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
  struct DownloaderTaskMetadata {
    DownloaderTaskMetadata() = delete;
    DownloaderTaskMetadata(FileDownloaderTask* task, event::RunnableAsyncTaskUPtr runnable)
        : task(task), runnable(std::move(runnable)), future(promise.get_future()) {}
    FileDownloaderTask* task;
    event::RunnableAsyncTaskUPtr runnable;
    // We use this promise/future to allow multiple concurrent downloads to the samefile
    // to create one downloader but track completion across the requesters.
    std::promise<Status> promise;
    std::shared_future<Status> future;
  };
  absl::Mutex downloader_mu_;
  absl::flat_hash_map<sole::uuid, std::unique_ptr<DownloaderTaskMetadata>> downloaders_
      ABSL_GUARDED_BY(downloader_mu_);
  std::string sha256sum_;
};

}  // namespace gml::gem::controller
