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

namespace gml::gem::fakegem {

using gml::internal::api::core::v1::EdgeCPMessage;

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

template <typename T>
std::unique_ptr<google::protobuf::Message> TryUnpack(const google::protobuf::Any& any) {
  if (any.Is<T>()) {
    auto ptr = std::make_unique<T>();
    any.UnpackTo(ptr.get());
    return ptr;
  }
  return nullptr;
}

std::unique_ptr<google::protobuf::Message> UnpackAny(const google::protobuf::Any& any) {
  if (auto ptr = TryUnpack<internal::api::core::v1::ImageOverlayChunk>(any)) return ptr;
  if (auto ptr = TryUnpack<internal::api::core::v1::H264Chunk>(any)) return ptr;
  if (auto ptr = TryUnpack<internal::api::core::v1::DeviceCapabilities>(any)) return ptr;
  if (auto ptr = TryUnpack<internal::api::core::v1::ExecutionGraphStatusUpdate>(any)) return ptr;
  if (auto ptr = TryUnpack<internal::api::core::v1::EdgeOTelMetrics>(any)) return ptr;
  LOG(FATAL) << "Unknown type: " << any.type_url();
  return nullptr;
}

std::vector<StreamData> ConvertEdgeCPMessageToStreamData(const std::vector<EdgeCPMessage>& msgs) {
  std::vector<StreamData> data;
  data.reserve(msgs.size());
  for (const auto& msg : msgs) {
    auto any = UnpackAny(msg.msg());
    data.push_back({
        .topic = msg.metadata().topic(),
        .msg = std::move(any),
        .timestamp_ns = msg.metadata().recv_timestamp().nanos() +
                        msg.metadata().recv_timestamp().seconds() * 1000 * 1000 * 1000,
        .is_otel = msg.msg().Is<internal::api::core::v1::EdgeOTelMetrics>(),
    });
  }
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
  data_ = std::make_unique<AllStreamsData>();
  GML_ASSIGN_OR_EXIT(data_->model_running_stream, LoadRecordedStream(blob_store_, stream_id_));
};

}  // namespace gml::gem::fakegem
