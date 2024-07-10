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

#include <Argus/Argus.h>
#include <sole.hpp>

namespace gml::gem::devices::argus {

// We are chosing an arbitrary layout mapping between argus UUIDs and sole UUIDs.
// The chosen mapping is:
// time_low | time_mid | time_hi_and_version => high bits
// clock_seq | node_0 | node_1 | node_2 | node_3 | node_4 | node_5 => low bits

inline sole::uuid ParseUUID(const Argus::UUID& uuid) {
  uint64_t high = 0;
  uint64_t low = 0;

  high |= uuid.time_low;

  high = high << 16;
  high |= uuid.time_mid;

  high = high << 16;
  high |= uuid.time_hi_and_version;

  low |= uuid.clock_seq;

  for (size_t i = 0; i < sizeof(uuid.node) / sizeof(char); i++) {
    low = low << 8;
    low |= uuid.node[i];
  }

  return sole::rebuild(high, low);
}

inline Argus::UUID ToArgusUUID(const sole::uuid& uuid) {
  Argus::UUID out;
  out.time_low = uuid.ab >> 32;
  out.time_mid = uuid.ab >> 16 & 0xFFFF;
  out.time_hi_and_version = uuid.ab & 0xFFFF;

  out.clock_seq = uuid.cd >> 48;

  uint64_t nodes = uuid.cd;
  for (size_t i = 0; i < sizeof(out.node) / sizeof(char); i++) {
    out.node[5 - i] = nodes & 0xFF;
    nodes = nodes >> 8;
  }
  return out;
}

}  // namespace gml::gem::devices::argus
