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

#include <memory>

#include "src/common/base/macros.h"
#include "src/common/system/cpu_info_reader.h"

namespace gml::system {

/**
 * LinuxCPUInfoReader is a linux specific implementation of the CPUInfoReader interface.
 * */
class LinuxCPUInfoReader : public CPUInfoReader {
 public:
  LinuxCPUInfoReader() = default;
  ~LinuxCPUInfoReader() override = default;
  Status Init() override;
  Status ReadCPUFrequencies(std::vector<CPUFrequencyInfo>* cpu_freqs) const override;
};

Status LinuxCPUInfoReader::Init() { return Status::OK(); }

Status LinuxCPUInfoReader::ReadCPUFrequencies(std::vector<CPUFrequencyInfo>* cpu_freqs) const {
  GML_UNUSED(cpu_freqs);
  return Status::OK();
}

std::unique_ptr<CPUInfoReader> CPUInfoReader::Create() {
  return std::unique_ptr<CPUInfoReader>(new LinuxCPUInfoReader());
}
}  // namespace gml::system
