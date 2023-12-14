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
struct AllStreamsData {
  std::vector<StreamData> model_running_stream;
};

class DownloadDataTask : public event::AsyncTask {
 public:
  DownloadDataTask(controller::CachedBlobStore* blob_store, std::string stream_id,
                   std::function<void(std::unique_ptr<AllStreamsData>)> download_complete)
      : blob_store_(blob_store),
        stream_id_(std::move(stream_id)),
        download_complete_(std::move(download_complete)) {}
  void Work() override;
  void Done() override { download_complete_(std::move(data_)); }

 private:
  std::vector<StreamData> data;
  std::unique_ptr<AllStreamsData> data_ = nullptr;
  controller::CachedBlobStore* blob_store_;
  std::string stream_id_;
  std::function<void(std::unique_ptr<AllStreamsData>)> download_complete_;
};

}  // namespace gml::gem::fakegem
