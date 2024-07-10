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

#include "src/common/system/fdinfo.h"

#include <fstream>
#include <memory>

#include "src/common/base/error.h"

namespace gml::system {

StatusOr<std::unique_ptr<FDInfoExtension>> FDInfoExtension::Parse(std::ifstream* in) {
  std::streamoff pos = in->tellg();

  auto drm_info_or_s = DRMFDInfo::Parse(in);
  if (drm_info_or_s.ok()) {
    return drm_info_or_s.ConsumeValueOrDie();
  }

  in->seekg(pos);
  // TODO(james): parse other extension types.

  return std::unique_ptr<FDInfoExtension>{};
}

StatusOr<std::unique_ptr<FDInfoExtension>> DRMFDInfo::Parse(std::ifstream* in) {
  auto fdinfo_ext = std::unique_ptr<FDInfoExtension>(new DRMFDInfo());
  auto* drm_info = static_cast<DRMFDInfo*>(fdinfo_ext.get());

  std::string line;
  while (std::getline(*in, line)) {
    std::vector<std::string_view> split = absl::StrSplit(line, absl::MaxSplits(':', 1));

    if (split.size() < 2) {
      return error::Internal("failed to parse fdinfo file");
    }

    std::string_view key = split[0];
    std::string_view val = absl::StripAsciiWhitespace(split[1]);

    if (!absl::StartsWith(key, "drm-")) {
      return error::InvalidArgument("input not a fdinfo DRM extension");
    }

    if (key == "drm-driver") {
      drm_info->driver_ = val;
    } else if (key == "drm-pdev") {
      drm_info->pdev_ = val;
    } else if (key == "drm-client-id") {
      if (!absl::SimpleAtoi(val, &drm_info->client_id_)) {
        return error::Internal("failed to parse drm-client-id");
      }
    } else if (absl::StartsWith(key, "drm-engine-capacity-")) {
      auto engine_name = absl::StripPrefix(key, "drm-engine-capacity-");
      if (!absl::SimpleAtoi(val, &drm_info->engines_[engine_name].capacity)) {
        return error::Internal("failed to parse $0", key);
      }
    } else if (absl::StartsWith(key, "drm-engine-")) {
      auto engine_name = absl::StripPrefix(key, "drm-engine-");
      val = absl::StripSuffix(val, " ns");
      if (!absl::SimpleAtoi(val, &drm_info->engines_[engine_name].busy_ns)) {
        return error::Internal("failed to parse $0", key);
      }
    }
  }
  if (drm_info->driver_ == "") {
    return error::InvalidArgument("input not a fdinfo DRM extension");
  }

  return fdinfo_ext;
}

std::string DRMFDInfo::ToString() const {
  std::vector<std::string> engine_strs;
  for (const auto& [name, engine] : engines_) {
    engine_strs.push_back(absl::Substitute("$0=$1", name, engine));
  }
  return absl::Substitute("DRMFDInfo{driver=$0, pdev=$1, client-id=$2, engines=$3}", driver_, pdev_,
                          client_id_, absl::StrJoin(engine_strs, ","));
}

Status ParseFDInfo(std::ifstream* in, FDInfo* fdinfo) {
  std::string line;
  while (std::getline(*in, line)) {
    std::vector<std::string_view> split = absl::StrSplit(line, absl::MaxSplits(':', 1));

    if (split.size() < 2) {
      return error::Internal("failed to parse fdinfo file");
    }

    std::string_view key = split[0];
    std::string_view val = absl::StripAsciiWhitespace(split[1]);

    if (key == "pos") {
      if (!absl::SimpleAtoi(val, &fdinfo->pos)) {
        return error::Internal("failed to parse pos from fdinfo file");
      }
    } else if (key == "flags") {
      if (!absl::SimpleAtoi(val, &fdinfo->flags)) {
        return error::Internal("failed to parse flags from fdinfo file");
      }
    } else if (key == "mnt_id") {
      if (!absl::SimpleAtoi(val, &fdinfo->mnt_id)) {
        return error::Internal("failed to parse mnt_id from fdinfo file");
      }
    } else if (key == "ino") {
      if (!absl::SimpleAtoi(val, &fdinfo->inode)) {
        return error::Internal("failed to parse ino from fdinfo file");
      }
      // "ino" should be the last key before the extensions.
      break;
    }
  }
  GML_ASSIGN_OR_RETURN(fdinfo->ext, FDInfoExtension::Parse(in));

  return Status::OK();
}

}  // namespace gml::system
