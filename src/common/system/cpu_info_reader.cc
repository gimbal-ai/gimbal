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

#include <cstdint>
#include <memory>

#include "src/common/base/macros.h"
#include "src/common/system/config.h"
#include "src/common/system/cpu_info_reader.h"
#include "src/common/system/parsers.h"

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

 private:
  std::string sysfs_path_;
};

Status LinuxCPUInfoReader::Init() {
  sysfs_path_ = ::gml::system::Config::GetInstance().sysfs_path();
  return Status::OK();
}

Status LinuxCPUInfoReader::ReadCPUFrequencies(std::vector<CPUFrequencyInfo>* cpu_freqs) const {
  CHECK_NOTNULL(cpu_freqs);
  // TODO(zasgar): We should add a fallback to read /proc/cpuinfo if sys does not work.
  int cpu_num = 0;
  while (true) {
    CPUFrequencyInfo finfo;
    finfo.cpu_num = cpu_num;
    auto s =
        ReadValueFromFile(absl::Substitute("$0/devices/system/cpu/cpu$1/cpufreq/scaling_cur_freq",
                                           sysfs_path_, cpu_num),
                          &finfo.cpu_freq_hz);
    if (!s.ok()) {
      if (cpu_num == 0) {
        return s;
      }
      // We probably finished reading all the files.
      return Status::OK();
    }

    // To get it in hertz.
    finfo.cpu_freq_hz *= static_cast<int64_t>(1000);
    cpu_freqs->emplace_back(finfo);
    cpu_num++;
  }
  return Status::OK();
}

std::unique_ptr<CPUInfoReader> CPUInfoReader::Create() {
  return std::unique_ptr<CPUInfoReader>(new LinuxCPUInfoReader());
}
}  // namespace gml::system
