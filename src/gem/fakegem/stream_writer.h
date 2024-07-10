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
  Status SendStreamData(const StreamDataWithOffset& data);
  bool IsVideoStreamRunning() { return video_running_.load(); }

  Status StartVideoStream();
  Status Stop();

 private:
  controller::GRPCBridge* bridge_;
  controller::CachedBlobStore* blob_store_;
  event::Dispatcher* dispatcher_;
  event::RunnableAsyncTaskUPtr download_data_task_ = nullptr;
  DataReplayer data_;

  // video_running_ tracks whether the client requests the video stream.
  // Not currently in use, but in the future, we might want to drop the video stream
  // messages while we're in the model running state.
  // For now if you have many GEMs running this will cause a lot of load on the controlplane.
  std::atomic<bool> video_running_ = false;
  event::TimerUPtr replay_timer_ = nullptr;

  sole::uuid pipeline_id_ = {.ab = 0, .cd = 0};
};
}  // namespace gml::gem::fakegem
