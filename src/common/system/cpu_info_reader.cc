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

#include "src/common/system/cpu_info_reader.h"

#include <cstdint>
#include <memory>

#include "src/common/base/macros.h"
#include "src/common/system/config.h"
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
  std::string sys_path_;
};

Status LinuxCPUInfoReader::Init() {
  sys_path_ = ::gml::system::Config::GetInstance().sys_path();
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
                                           sys_path_, cpu_num),
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

StatusOr<std::unique_ptr<CPUInfoReader>> CPUInfoReader::Create() {
  auto reader = new LinuxCPUInfoReader();
  GML_RETURN_IF_ERROR(reader->Init());
  return std::unique_ptr<CPUInfoReader>(reader);
}
}  // namespace gml::system
