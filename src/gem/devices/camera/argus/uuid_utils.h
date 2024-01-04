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
