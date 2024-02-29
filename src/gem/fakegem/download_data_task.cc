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

#include "src/gem/fakegem/download_data_task.h"

#include <fstream>

#include <google/protobuf/util/delimited_message_util.h>

#include "src/api/corepb/v1/cp_edge.pb.h"
#include "src/api/corepb/v1/mediastream.pb.h"

DEFINE_string(model_idle_stream, gflags::StringFromEnv("GML_MODEL_IDLE_STREAM", ""),
              "The id:sha256sum:size of the model idle stream. This is when the device is not "
              "running a model yet. It will be looped over until the state changes.");
DEFINE_string(model_compiling_stream, gflags::StringFromEnv("GML_MODEL_COMPILING_STREAM", ""),
              "The id:sha256sum:size of the model compiling stream. This is when the device is "
              "transitioning from idle to running. It does not loop.");
DEFINE_string(model_running_stream, gflags::StringFromEnv("GML_MODEL_RUNNING_STREAM", ""),
              "The id:sha256sum:size of the model running stream. This is what happens when the "
              "device is running a model. It will be looped over until the state changes. Ie "
              "'6e6ee5ae-a795-4e88-9e92-0ddce60da93b:"
              "9831bd13284280438f0988ef90fe9208be4b64519221b229c7aeb592f85d0ede:295755'");
namespace gml::gem::fakegem {
using gml::internal::api::core::v1::EdgeCPMessage;

// The duration we sleep for if we can't figure out the next timestamp, such as
// when we are at the end of the stream. 1/25th of a second. Arbitrarily decided to be
// the same as the period between frames in a 25fps video.
const int64_t kFallbackSleepDurationNs = 1000 * 1000 * 1000 / 25;

void DataReplayer::SetDesiredStreamState(StreamState state) {
  absl::MutexLock lock(&current_state_lock_);
  if (state == current_state_) {
    LOG(INFO) << "Ignore desired state call as we already are in state "
              << magic_enum::enum_name(state);
    return;
  }
  LOG(INFO) << "Setting desired state to " << magic_enum::enum_name(state);
  current_state_ = state;
  data_index_ = 0;
}

const std::vector<StreamData>& DataReplayer::GetStateStream(StreamState stream_state) {
  switch (stream_state) {
    case StreamState::kModelIdle:
      return replay_data_->model_idle_stream_;
    case StreamState::kModelCompiling:
      return replay_data_->model_compiling_stream_;
    case StreamState::kModelRunning:
      return replay_data_->model_running_stream_;
  }
}

StreamDataWithOffset DataReplayer::Next() {
  size_t data_index;
  StreamState state;
  {
    absl::MutexLock lock(&current_state_lock_);
    if (data_index_ >= GetStateStream(current_state_).size()) {
      current_state_ = NextMapState(current_state_);
      LOG(INFO) << absl::Substitute("Transition to next state: $0",
                                    magic_enum::enum_name(current_state_));
      data_index_ = 0;
    }
    state = current_state_;
    data_index = data_index_;
    data_index_++;
  }

  const auto& current_stream = GetStateStream(state);
  if (data_index == 0) {
    uint64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          std::chrono::system_clock::now().time_since_epoch())
                          .count();

    timestamp_offset_ns_ = now_ns - current_stream[0].timestamp_ns;
  }
  const StreamData& cur_msg = current_stream[data_index++];
  int64_t sleep_time =
      current_stream[data_index % current_stream.size()].timestamp_ns - cur_msg.timestamp_ns;
  return {
      .data = cur_msg,
      .ts_offset_ns = timestamp_offset_ns_,
      .sleep_for = sleep_time < 0 ? kFallbackSleepDurationNs : static_cast<uint64_t>(sleep_time)};
}

template <typename T>
gml::StatusOr<std::vector<T>> ReadMessagesFromFile(const std::filesystem::path& path) {
  std::ifstream ifs(path, std::ifstream::in | std::ifstream::binary);
  if (!ifs.good()) {
    return gml::error::Internal("Failed to read file $0 ($1)", path.generic_string(),
                                strerror(errno));
  }
  google::protobuf::io::IstreamInputStream zstream(&ifs);
  std::vector<T> chunks;
  while (true) {
    T chunk;
    bool cleaneof = true;
    bool read =
        google::protobuf::util::ParseDelimitedFromZeroCopyStream(&chunk, &zstream, &cleaneof);
    if (!read) {
      if (!cleaneof) {
        LOG(INFO) << "Not clean eof";
      }
      break;
    }
    chunks.push_back(chunk);
  }

  return chunks;
}

class MessageStats {
 public:
  void AddInstance(const std::string& type_url) { counts[type_url]++; }
  /**
   * Returns the count of each type url in a string, sorted alphabetically.
   */
  std::string ToString() const {
    std::vector<std::string> out;
    for (const auto& [type_url, count] : counts) {
      out.push_back(absl::StrCat(type_url, ": ", count));
    }
    return absl::StrJoin(out, "\n");
  }

 private:
  absl::flat_hash_map<std::string, int64_t> counts;
};

template <typename T>
std::unique_ptr<google::protobuf::Message> TryUnpack(const google::protobuf::Any& any,
                                                     MessageStats* stats) {
  if (any.Is<T>()) {
    auto ptr = std::make_unique<T>();
    stats->AddInstance(ptr->GetTypeName());
    any.UnpackTo(ptr.get());
    return ptr;
  }
  return nullptr;
}

std::unique_ptr<google::protobuf::Message> UnpackAny(const google::protobuf::Any& any,
                                                     MessageStats* stats) {
  if (auto ptr = TryUnpack<internal::api::core::v1::ImageOverlayChunk>(any, stats)) return ptr;
  if (auto ptr = TryUnpack<internal::api::core::v1::H264Chunk>(any, stats)) return ptr;
  if (auto ptr = TryUnpack<internal::api::core::v1::DeviceCapabilities>(any, stats)) return ptr;
  if (auto ptr = TryUnpack<internal::api::core::v1::ExecutionGraphStatusUpdate>(any, stats)) {
    return ptr;
  }
  if (auto ptr = TryUnpack<internal::api::core::v1::VideoHeader>(any, stats)) {
    return ptr;
  }
  if (auto ptr = TryUnpack<internal::api::core::v1::EdgeOTelMetrics>(any, stats)) return ptr;
  LOG(FATAL) << "Unknown type: " << any.type_url();
  return nullptr;
}

std::vector<StreamData> ConvertEdgeCPMessageToStreamData(const std::vector<EdgeCPMessage>& msgs) {
  std::vector<StreamData> data;
  data.reserve(msgs.size());
  MessageStats stats;
  for (const auto& msg : msgs) {
    auto any = UnpackAny(msg.msg(), &stats);
    data.push_back({
        .topic = msg.metadata().topic(),
        .msg = std::move(any),
        .timestamp_ns = msg.metadata().recv_timestamp().nanos() +
                        msg.metadata().recv_timestamp().seconds() * 1000 * 1000 * 1000,
        .is_otel = msg.msg().Is<internal::api::core::v1::EdgeOTelMetrics>(),
    });
  }
  LOG(INFO) << "Message stats: \n" << stats.ToString();
  return data;
}

StatusOr<std::vector<StreamData>> LoadRecordedStream(controller::CachedBlobStore* blob_store,
                                                     const std::string& stream_id) {
  std::vector<std::string> parts = absl::StrSplit(stream_id, ':');
  if (parts.size() != 3) {
    return gml::error::Internal("Invalid model running stream flag: $0", stream_id);
  }
  std::string id = parts[0];
  std::string sha256sum = parts[1];
  size_t bytes;
  bool ok = absl::SimpleAtoi(parts[2], &bytes);
  if (!ok) {
    return gml::error::Internal("Invalid model running stream flag: $0 last part is not a number",
                                stream_id);
  }
  LOG(INFO) << "Downloading file with id: " << id;
  GML_RETURN_IF_ERROR(blob_store->EnsureBlobExists(id, sha256sum, bytes));

  GML_ASSIGN_OR_RETURN(auto path, blob_store->FilePath(id));
  LOG(INFO) << absl::Substitute("File download '$0' complete. Reading chunks.", id);
  GML_ASSIGN_OR_EXIT(auto recorded_messages, ReadMessagesFromFile<EdgeCPMessage>(path));
  return ConvertEdgeCPMessageToStreamData(recorded_messages);
}

void DownloadDataTask::Work() {
  LOG(INFO) << "Downloading model running stream";
  GML_ASSIGN_OR_EXIT(auto model_running_stream,
                     LoadRecordedStream(blob_store_, FLAGS_model_running_stream));
  LOG(INFO) << "Downloading model idle stream";
  GML_ASSIGN_OR_EXIT(auto model_idle_stream,
                     LoadRecordedStream(blob_store_, FLAGS_model_idle_stream));
  LOG(INFO) << "Downloading model compiling stream";
  GML_ASSIGN_OR_EXIT(auto model_compiling_stream,
                     LoadRecordedStream(blob_store_, FLAGS_model_compiling_stream));
  data_ =
      std::make_unique<ReplayData>(std::move(model_running_stream), std::move(model_idle_stream),
                                   std::move(model_compiling_stream));
};

}  // namespace gml::gem::fakegem
