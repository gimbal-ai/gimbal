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

#include <unistd.h>

#include <chrono>
#include <fstream>
#include <utility>

#include <google/protobuf/any.pb.h>
#include <google/protobuf/util/delimited_message_util.h>
#include <grpcpp/grpcpp.h>
#include <sole.hpp>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/controller/cached_blob_store.h"
#include "src/gem/fakegem/download_data_task.h"

namespace gml::gem::fakegem {
class StreamWriter {
 public:
  StreamWriter(controller::GRPCBridge* bridge, controller::CachedBlobStore* blob_store,
               event::Dispatcher* dispatcher)
      : bridge_(bridge), blob_store_(blob_store), dispatcher_(dispatcher) {}
  Status Run();
  bool IsPipelineRunning() { return pipeline_id_.ab != 0 && pipeline_id_.cd != 0; }

  Status StartModelStream(sole::uuid id);
  Status SendStreamData(const StreamData& data);
  bool IsVideoStreamRunning() { return video_running_.load(); }

  Status StartVideoStream();
  Status Stop();

 private:
  controller::GRPCBridge* bridge_;
  controller::CachedBlobStore* blob_store_;
  event::Dispatcher* dispatcher_;
  event::RunnableAsyncTaskUPtr download_data_task_ = nullptr;
  std::unique_ptr<AllStreamsData> data_ = nullptr;

  // video_running_ tracks whether the client requests the video stream.
  // Not currently in use, but in the future, we might want to drop the video stream
  // messages while we're in the model running state.
  // For now if you have many GEMs running this will cause a lot of load on the controlplane.
  std::atomic<bool> video_running_ = false;
  size_t data_index_ = 0;
  event::TimerUPtr replay_timer_ = nullptr;
  // This is the time_offset_ that we add to the saved times to match the current time.
  uint64_t timestamp_offset_ns_ = 0;

  sole::uuid pipeline_id_ = {.ab = 0, .cd = 0};
};
}  // namespace gml::gem::fakegem
