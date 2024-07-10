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

#include "src/common/system/nvmap.h"

#include <fstream>
#include <memory>

#include <gmock/gmock.h>

#include "src/common/base/status.h"
#include "src/common/bazel/runfiles.h"
#include "src/common/testing/status.h"
#include "src/common/testing/testing.h"

namespace gml::system {

constexpr std::string_view kIOVMMPath = "src/common/system/testdata/sys/kernel/debug/nvmap/iovmm";

bool operator==(const IOVMMClient& a, const IOVMMClient& b) {
  return (a.client_type == b.client_type) && (a.cmdline == b.cmdline) && (a.pid == b.pid) &&
         (a.size_bytes == b.size_bytes);
}

TEST(ParseNVMapIOVMMClients, ParsesTestDataCorrectly) {
  auto path = bazel::RunfilePath(std::filesystem::path(kIOVMMPath) / "clients");
  std::vector<IOVMMClient> clients;
  ASSERT_OK(ParseNVMapIOVMMClients(path, &clients));

  /**
   CLIENT                        PROCESS      PID        SIZE
   user                              gem     4481     118372K
   user                      gnome-shell     2831       5384K
   user                             Xorg     1731      18140K
   user                   nvargus-daemon     1012          0K
   total                                              141896K
  */
  EXPECT_THAT(clients, ::testing::UnorderedElementsAre(
                           IOVMMClient{
                               .client_type = IOVMMClient::IOVMM_CLIENT_TYPE_USER,
                               .cmdline = "gem",
                               .pid = 4481,
                               .size_bytes = 118372 * 1024,
                           },
                           IOVMMClient{
                               .client_type = IOVMMClient::IOVMM_CLIENT_TYPE_USER,
                               .cmdline = "gnome-shell",
                               .pid = 2831,
                               .size_bytes = 5384 * 1024,
                           },
                           IOVMMClient{
                               .client_type = IOVMMClient::IOVMM_CLIENT_TYPE_USER,
                               .cmdline = "Xorg",
                               .pid = 1731,
                               .size_bytes = 18140 * 1024,
                           },
                           IOVMMClient{
                               .client_type = IOVMMClient::IOVMM_CLIENT_TYPE_USER,
                               .cmdline = "nvargus-daemon",
                               .pid = 1012,
                               .size_bytes = 0,
                           },
                           IOVMMClient{
                               .client_type = IOVMMClient::IOVMM_CLIENT_TYPE_TOTAL,
                               .cmdline = "",
                               .pid = 0,
                               .size_bytes = 141896 * 1024,
                           }));
}

TEST(ParseNVMapIOVMMClients, ParseFailsOnBadFile) {
  auto path = bazel::RunfilePath(std::filesystem::path(kIOVMMPath) / "clients_bad");
  std::vector<IOVMMClient> clients;
  auto s = ParseNVMapIOVMMClients(path, &clients);
  EXPECT_FALSE(s.ok());
  EXPECT_EQ(types::Code::CODE_INTERNAL, s.code());
}

}  // namespace gml::system
