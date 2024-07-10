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

#include <utility>

#include <sole.hpp>

#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/gem/exec/core/context.h"

namespace gml::gem::exec::core {

/**
 * ControlExecutionContext allows calculators to interface with GEM's controller.
 * */
class ControlExecutionContext : public ExecutionContext {
 public:
  using VideoWithOverlaysCallback =
      std::function<Status(const std::vector<std::unique_ptr<google::protobuf::Message>>&)>;

  void RegisterVideoWithOverlaysCallback(VideoWithOverlaysCallback cb) {
    absl::base_internal::SpinLockHolder lock(&video_cb_lock_);
    video_w_overlays_cb_ = std::move(cb);
  }

  void ClearVideoWithOverlaysCallback() {
    absl::base_internal::SpinLockHolder lock(&video_cb_lock_);
    video_w_overlays_cb_ = nullptr;
  }

  VideoWithOverlaysCallback GetVideoWithOverlaysCallback() {
    absl::base_internal::SpinLockHolder lock(&video_cb_lock_);
    return video_w_overlays_cb_;
  }

  void SetLogicalPipelineID(const sole::uuid& logical_pipeline_id) {
    absl::base_internal::SpinLockHolder lock(&logical_pipeline_id_lock_);
    logical_pipeline_id_ = logical_pipeline_id;
  }

  sole::uuid GetLogicalPipelineID() {
    absl::base_internal::SpinLockHolder lock(&logical_pipeline_id_lock_);
    return logical_pipeline_id_;
  }

 private:
  absl::base_internal::SpinLock video_cb_lock_;
  VideoWithOverlaysCallback video_w_overlays_cb_ ABSL_GUARDED_BY(video_cb_lock_);

  absl::base_internal::SpinLock logical_pipeline_id_lock_;
  sole::uuid logical_pipeline_id_ ABSL_GUARDED_BY(logical_pipeline_id_lock_);
};

}  // namespace gml::gem::exec::core
