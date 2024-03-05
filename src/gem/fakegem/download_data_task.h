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

#include <utility>

#include "src/common/base/base.h"
#include "src/gem/controller/cached_blob_store.h"

namespace gml::gem::fakegem {

struct StreamData {
  internal::api::core::v1::EdgeCPTopic topic;
  std::unique_ptr<google::protobuf::Message> msg;
  int64_t timestamp_ns;
  bool is_otel;
};
struct StreamDataWithOffset {
  const StreamData& data;
  uint64_t ts_offset_ns;
  uint64_t sleep_for;
};

enum class StreamState { kModelIdle, kModelRunning };

struct ReplayData {
  ReplayData(std::vector<StreamData> model_running_stream,
             std::vector<StreamData> model_idle_stream)
      : model_running_stream_(std::move(model_running_stream)),
        model_idle_stream_(std::move(model_idle_stream)) {}

  std::vector<StreamData> model_running_stream_;
  std::vector<StreamData> model_idle_stream_;
};

class DataReplayer {
 public:
  ~DataReplayer() = default;
  StreamDataWithOffset Next();

  void SetReplayData(std::unique_ptr<ReplayData> replay_data) {
    replay_data_ = std::move(replay_data);
  }
  bool HasData() { return replay_data_ != nullptr; }

  void SetDesiredStreamState(StreamState state);
  StreamState current_state() {
    absl::ReaderMutexLock lock(&current_state_lock_);
    return current_state_;
  }
  const std::vector<StreamData>& GetStateStream(StreamState stream_state);

 private:
  absl::Mutex current_state_lock_;
  StreamState current_state_ ABSL_GUARDED_BY(current_state_lock_) = StreamState::kModelIdle;
  size_t data_index_ ABSL_GUARDED_BY(current_state_lock_) = 0;
  std::unique_ptr<ReplayData> replay_data_;

  // This is the time_offset_ that we add to the saved times to match the current time.
  uint64_t timestamp_offset_ns_ = 0;
};

class DownloadDataTask : public event::AsyncTask {
 public:
  DownloadDataTask(controller::CachedBlobStore* blob_store,
                   std::function<void(std::unique_ptr<ReplayData>)> download_complete)
      : blob_store_(blob_store), download_complete_(std::move(download_complete)) {}
  void Work() override;
  void Done() override { download_complete_(std::move(data_)); }

 private:
  std::vector<StreamData> data;
  std::unique_ptr<ReplayData> data_ = nullptr;
  controller::CachedBlobStore* blob_store_;
  std::function<void(std::unique_ptr<ReplayData>)> download_complete_;
};

}  // namespace gml::gem::fakegem
