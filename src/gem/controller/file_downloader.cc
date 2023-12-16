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

#include "src/gem/controller/file_downloader.h"

#include <fcntl.h>

#include <cerrno>
#include <future>
#include <memory>
#include <queue>
#include <utility>

#include "src/common/base/base.h"
#include "src/common/base/status.h"
#include "src/common/event/task.h"
#include "src/common/fs/fs_utils.h"
#include "src/common/fs/fs_wrapper.h"
#include "src/common/fs/temp_file.h"
#include "src/common/grpcutils/status.h"
#include "src/common/uuid/uuid_utils.h"
#include "src/controlplane/egw/egwpb/v1/egwpb.pb.h"
#include "src/controlplane/filetransfer/ftpb/v1/ftpb.pb.h"

using ::gml::internal::api::core::v1::FileTransferRequest;
using ::gml::internal::api::core::v1::FileTransferResponse;

namespace gml::gem::controller {
const size_t kChunkSize = 64ULL * 1024ULL;
const int kParallelChunks = 32;
const absl::Duration kRequestTimeout = absl::Milliseconds(2000);

// Metadata corresponding to each request that has happened.
struct RequestMetadata {
  absl::Time req_time;
  size_t start_pos;
  size_t size;
};

class FileDownloaderTask : public event::AsyncTask {
 public:
  using DownloadCompleteCallback = std::function<void(Status, sole::uuid, std::filesystem::path)>;
  using MsgQueue = std::queue<std::unique_ptr<FileTransferResponse>>;
  FileDownloaderTask() = delete;
  FileDownloaderTask(GRPCBridge* bridge, const sole::uuid& fid, int size,
                     std::string_view sha256sum, std::filesystem::path file_path,
                     DownloadCompleteCallback cb = nullptr)
      : bridge_(bridge),
        fid_(fid),
        size_(size),
        expected_sha256sum_(sha256sum),
        download_complete_callback_(std::move(cb)),
        file_path_(std::move(file_path)),
        fd_(-1),
        current_file_pos_(0) {}

  void HandleFileTransferResponse(std::unique_ptr<FileTransferResponse> resp);
  Status RequestChunk(RequestMetadata* md);

  Status Init();
  Status WorkImpl();
  void Work() override { download_status_ = WorkImpl(); }
  void Done() override;
  void Stop();

  Status GenerateRequests();

 private:
  GRPCBridge* bridge_;
  sole::uuid fid_;
  size_t size_;

  std::atomic<bool> running_ = true;

  // These variables are only updated by the primary thread running Work().
  Status download_status_;
  std::string expected_sha256sum_;
  DownloadCompleteCallback download_complete_callback_;
  std::filesystem::path file_path_;

  int fd_;
  // Store the file position that we have gotten to. When this reaches size_, we don't need to
  // make any more requests. These variables should only be accesses in the Work() thread.
  size_t current_file_pos_;
  // Map from start_pos : RequestMetadata.
  absl::flat_hash_map<size_t, RequestMetadata> outstanding_requests_;

  // We store messages as they arrive here and use the ABSL mutexes to signal when data is
  // avaialble.
  absl::Mutex msgs_mu_;
  MsgQueue msgs_ ABSL_GUARDED_BY(msgs_mu_);
};

FileDownloader::FileDownloader(event::Dispatcher* d, GEMInfo* agent_info, GRPCBridge* bridge)
    : MessageHandler(d, agent_info, bridge) {}

Status FileDownloader::Init() { return Status::OK(); }

Status FileDownloader::BlockingDownload(const sole::uuid& fid, const std::string& sha256sum,
                                        size_t size, const std::filesystem::path& new_path) {
  auto download_complete_handler = [&](Status status, sole::uuid id,
                                       const std::filesystem::path& path) {
    DEFER({
      absl::MutexLock l(&downloader_mu_);
      auto it = downloaders_.find(fid);
      if (it == downloaders_.end()) {
        return;
      }
      auto downloader_md = downloaders_.extract(it);
      downloader_md.mapped()->promise.set_value(status);
      dispatcher()->DeferredDelete(std::move(downloader_md.mapped()->runnable));
    });

    LOG(INFO) << absl::Substitute("File download complete, fid: $0, status: $1\n", id.str(),
                                  status.ToString());

    if (!status.ok()) {
      return;
    }
    // Try atomic rename of file.
    std::error_code ec;
    std::filesystem::rename(path, new_path, ec);
    if (ec) {
      LOG(ERROR) << "Atomic rename of file: \"" << path << "\" failed, will copy instead";
      status = fs::Copy(path, new_path, std::filesystem::copy_options::overwrite_existing);
      if (!status.ok()) {
        return;
      }
    }
  };

  std::filesystem::path download_path = new_path;
  download_path += ".tmp";
  std::shared_future<Status> barrier;
  {
    absl::MutexLock l(&downloader_mu_);
    auto it = downloaders_.find(fid);
    if (it == downloaders_.end()) {
      // Insert into the downloaders.
      auto downloader = std::make_unique<FileDownloaderTask>(
          bridge(), fid, size, sha256sum, download_path, download_complete_handler);
      auto md = std::make_unique<DownloaderTaskMetadata>(
          downloader.get(), dispatcher()->CreateAsyncTask(std::move(downloader)));
      md->runnable->Run();
      barrier = md->future;
      downloaders_[fid] = std::move(md);
    } else {
      barrier = it->second->future;
    }
  }

  return barrier.get();
}

Status FileDownloader::HandleMessage(
    const gml::internal::controlplane::egw::v1::BridgeResponse& msg) {
  auto resp = std::make_unique<FileTransferResponse>();
  if (!msg.msg().UnpackTo(resp.get())) {
    LOG(ERROR) << "Failed to unpack file transfer response message. Received message of type: "
               << msg.msg().type_url() << " . Ignoring...";
    return Status::OK();
  }

  auto fid = ParseUUID(resp->file_id());
  downloader_mu_.Lock();
  auto it = downloaders_.find(fid);
  auto end = downloaders_.end();
  downloader_mu_.Unlock();
  if (it != end) {
    it->second->task->HandleFileTransferResponse(std::move(resp));
  }
  // Downloader has already been removed, this is likely a delayed response from the controlplane.
  return Status::OK();
}

Status FileDownloader::Finish() {
  absl::MutexLock l(&downloader_mu_);
  for (auto& m : downloaders_) {
    m.second->task->Stop();
  }
  return Status::OK();
}

void FileDownloaderTask::HandleFileTransferResponse(std::unique_ptr<FileTransferResponse> resp) {
  absl::MutexLock l(&msgs_mu_);
  msgs_.emplace(std::move(resp));
}

Status FileDownloaderTask::RequestChunk(RequestMetadata* md) {
  CHECK_NOTNULL(md);
  FileTransferRequest req;
  ToProto(fid_, req.mutable_file_id());
  req.set_chunk_start_bytes(static_cast<int64_t>(md->start_pos));
  req.set_num_bytes(static_cast<int64_t>(md->size));
  md->req_time = absl::Now();

  return bridge_->SendMessageToBridge(internal::api::core::v1::EDGE_CP_TOPIC_FILE_TRANSFER, req);
}

Status FileDownloaderTask::Init() {
  current_file_pos_ = 0;
  download_status_ = Status::OK();
  // We use the c file system library here since want to use lseek to write the file
  // in chunks.
  fd_ = open(file_path_.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0775);
  if (fd_ < 0) {
    return error::FailedPrecondition("Failed to open file: $0", errno);
  }
  return Status::OK();
}

static bool QHasData(FileDownloaderTask::MsgQueue* q) { return !q->empty(); };

Status FileDownloaderTask::WorkImpl() {
  GML_RETURN_IF_ERROR(Init());
  while (running_) {
    GML_RETURN_IF_ERROR(GenerateRequests());
    std::unique_ptr<FileTransferResponse> resp;
    if (msgs_mu_.LockWhenWithTimeout(absl::Condition(QHasData, &msgs_), kRequestTimeout)) {
      // Lock held, has data.
      if (msgs_.front() == nullptr) {
        // Termination because nullptr is pushed onto the q.
        msgs_mu_.Unlock();
        return error::Cancelled("downloaded closed before completion");
      }
      resp = std::move(msgs_.front());
      msgs_.pop();
      msgs_mu_.Unlock();
    } else {
      // Lock held, but we got a timeout. We just continue and new requests will be made.
      msgs_mu_.Unlock();
      continue;
    }

    GML_RETURN_IF_ERROR(Status(resp->status()));

    size_t start_bytes = resp->chunk().start_bytes();
    auto it = outstanding_requests_.find(start_bytes);
    if (it == outstanding_requests_.end()) {
      // Duplicate response, we can ignore it.
      continue;
    }
    outstanding_requests_.erase(it);

    auto payload = resp->chunk().payload();
    size_t size = payload.size();

    // Write to File.
    lseek(fd_, static_cast<off_t>(start_bytes), SEEK_SET);
    auto byte_remain = static_cast<ssize_t>(size);
    ssize_t offset = 0;
    while (byte_remain > 0) {
      ssize_t num_bytes = write(fd_, payload.c_str() + offset, byte_remain);
      if (num_bytes < 0) {
        return error::Internal("failed to write file: $0", errno);
      }
      byte_remain -= num_bytes;
      offset += num_bytes;
    }

    if ((current_file_pos_ >= size_) && outstanding_requests_.empty()) {
      // Download is complete.
      break;
    }
  }

  close(fd_);

  GML_ASSIGN_OR_RETURN(std::string sha256sum_str, fs::GetSHA256Sum(file_path_));
  if (sha256sum_str != expected_sha256sum_) {
    return error::Unknown("expected file hash: $0, got $1", expected_sha256sum_, sha256sum_str);
  }
  return Status::OK();
}

void FileDownloaderTask::Done() {
  if (download_complete_callback_) {
    download_complete_callback_(download_status_, fid_, file_path_);
  }
}

void FileDownloaderTask::Stop() {
  // This function is called on an interruption and will not lead to the file being downloaded
  // fully.
  running_ = false;

  // Write a wake up to the reader thread.
  absl::MutexLock l(&msgs_mu_);
  msgs_.emplace(nullptr);
}

Status FileDownloaderTask::GenerateRequests() {
  auto now = absl::Now();
  // Check and see if any of the outstanding_requests have timed out and re-make them.
  for (auto& kv : outstanding_requests_) {
    if ((now - kv.second.req_time) > kRequestTimeout) {
      LOG(INFO) << "Re-requesting: " << kv.second.start_pos;
      GML_RETURN_IF_ERROR(RequestChunk(&kv.second));
    }
  }

  while (current_file_pos_ < size_ && outstanding_requests_.size() < kParallelChunks) {
    RequestMetadata md;
    md.start_pos = current_file_pos_;
    md.size = std::min(size_ - current_file_pos_, kChunkSize);
    current_file_pos_ += md.size;
    GML_RETURN_IF_ERROR(RequestChunk(&md));
    outstanding_requests_[md.start_pos] = md;
  }
  return Status::OK();
}
}  // namespace gml::gem::controller
