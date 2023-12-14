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

#include "src/gem/fakegem/stream_writer.h"

#include <unistd.h>

#include <chrono>
#include <fstream>
#include <utility>

#include <google/protobuf/any.pb.h>
#include <grpcpp/grpcpp.h>
#include <sole.hpp>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/api/corepb/v1/mediastream.pb.h"
#include "src/common/base/base.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.grpc.pb.h"
#include "src/gem/controller/cached_blob_store.h"

DEFINE_string(model_running_stream, gflags::StringFromEnv("GML_MODEL_RUNNING_STREAM", ""),
              "The id:sha256sum:size of the model running stream. Ie "
              "'6e6ee5ae-a795-4e88-9e92-0ddce60da93b:"
              "9831bd13284280438f0988ef90fe9208be4b64519221b229c7aeb592f85d0ede:295755'");

namespace gml::gem::fakegem {

// The duration we sleep for if we can't figure out the next timestamp, such as
// when we are at the end of the stream. 1/25th of a second. Arbitrarily decided to be
// the same as the period between frames in a 25fps video.
const int64_t kFallbackSleepDurationNs = 1000 * 1000 * 1000 / 25;

template <typename T>
void SetMetricTime(T* dp, uint64_t timestamp_offset_ns) {
  dp->set_time_unix_nano(dp->time_unix_nano() + timestamp_offset_ns);
  dp->set_start_time_unix_nano(dp->start_time_unix_nano() + timestamp_offset_ns);
}

internal::api::core::v1::EdgeOTelMetrics RewriteOTelTimestamps(
    internal::api::core::v1::EdgeOTelMetrics* otel_msg, uint64_t timestamp_offset_ns) {
  internal::api::core::v1::EdgeOTelMetrics updated_msg;
  updated_msg.CopyFrom(*otel_msg);
  for (auto& scope_metric : *updated_msg.mutable_resource_metrics()->mutable_scope_metrics()) {
    for (auto& metric : *scope_metric.mutable_metrics()) {
      switch (metric.data_case()) {
        case opentelemetry::proto::metrics::v1::Metric::kGauge: {
          for (auto& dp : *metric.mutable_gauge()->mutable_data_points()) {
            SetMetricTime(&dp, timestamp_offset_ns);
          }
          break;
        }
        case opentelemetry::proto::metrics::v1::Metric::kSum: {
          for (auto& dp : *metric.mutable_sum()->mutable_data_points()) {
            SetMetricTime(&dp, timestamp_offset_ns);
          }
          break;
        }
        case opentelemetry::proto::metrics::v1::Metric::kHistogram: {
          for (auto& dp : *metric.mutable_histogram()->mutable_data_points()) {
            SetMetricTime(&dp, timestamp_offset_ns);
          }
          break;
        }
        case opentelemetry::proto::metrics::v1::Metric::kExponentialHistogram: {
          for (auto& dp : *metric.mutable_exponential_histogram()->mutable_data_points()) {
            SetMetricTime(&dp, timestamp_offset_ns);
          }
          break;
        }
        case opentelemetry::proto::metrics::v1::Metric::kSummary: {
          for (auto& dp : *metric.mutable_summary()->mutable_data_points()) {
            SetMetricTime(&dp, timestamp_offset_ns);
          }
          break;
        }
        case opentelemetry::proto::metrics::v1::Metric::DATA_NOT_SET:
          break;
      }
    }
  }
  return updated_msg;
}

Status StreamWriter::Run() {
  replay_timer_ = dispatcher_->CreateTimer([this]() {
    if (data_ == nullptr) {
      LOG(FATAL) << "Data not loaded";
    }

    if (data_index_ == 0) {
      LOG(INFO) << "Starting replay from the beginning";
      uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();

      timestamp_offset_ns_ = now_ns - data_->model_running_stream[0].timestamp_ns;
    }

    const auto& cur_msg = data_->model_running_stream[data_index_];
    auto status = SendStreamData(cur_msg);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to send message to bridge: " << status.msg();
    }

    data_index_++;
    data_index_ %= data_->model_running_stream.size();
    int64_t sleep_time =
        data_->model_running_stream[data_index_].timestamp_ns - cur_msg.timestamp_ns;
    if (sleep_time < 0) {
      sleep_time = kFallbackSleepDurationNs;
    }
    if (!replay_timer_) {
      LOG(INFO) << "Run complete";
      return;
    }
    replay_timer_->EnableTimer(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::nanoseconds(sleep_time)));
  });
  // Run the download task in the dispatcher because the file downloader needs to
  // run in the event loop. If you call it outside, you can end up blocking the dispatcher from ever
  // starting, since StatusWriter::Run() is usually called before dispatcher_->Run().
  download_data_task_ = dispatcher_->CreateAsyncTask(std::make_unique<DownloadDataTask>(
      blob_store_, FLAGS_model_running_stream, [this](std::unique_ptr<AllStreamsData> data) {
        data_ = std::move(data);
        dispatcher_->DeferredDelete(std::move(download_data_task_));
        replay_timer_->EnableTimer(std::chrono::milliseconds(0));
      }));
  download_data_task_->Run();
  LOG(INFO) << "Timer enabled";
  return Status::OK();
};

Status StreamWriter::SendStreamData(const StreamData& data) {
  if (!data.is_otel) {
    return bridge_->SendMessageToBridge(data.topic, *data.msg);
  }
  // We need to rewrite the timestamps in the otel message.
  auto* otel_msg = static_cast<internal::api::core::v1::EdgeOTelMetrics*>(data.msg.get());
  if (!otel_msg) {
    LOG(FATAL) << "Failed to cast message to otel message";
  }
  return bridge_->SendMessageToBridge(data.topic,
                                      RewriteOTelTimestamps(otel_msg, timestamp_offset_ns_));
}

Status StreamWriter::StartModelStream(sole::uuid id) {
  if (!IsPipelineRunning()) {
    LOG(INFO) << "Starting pipeline << " << id.str();
    pipeline_id_ = id;
    return Status::OK();
  }
  if (IsPipelineRunning() && id != pipeline_id_) {
    LOG(INFO) << absl::Substitute("Pipeline $0 already running. Refusing to deploy pipeline $1",
                                  pipeline_id_.str(), id.str());
    return Status::OK();
  }
  return Status::OK();
}

Status StreamWriter::StartVideoStream() {
  video_running_.store(true);
  return Status::OK();
}
Status StreamWriter::Stop() {
  replay_timer_->DisableTimer();
  replay_timer_.reset();
  return Status::OK();
}
}  // namespace gml::gem::fakegem
