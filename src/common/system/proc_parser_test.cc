/*
 * Copyright 2018- The Pixie Authors.
 * Modifications Copyright 2023- Gimlet Labs, Inc.
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

#include "src/common/system/proc_parser.h"

#include <istream>
#include <memory>
#include <sstream>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "src/common/bazel/runfiles.h"
#include "src/common/fs/fs_wrapper.h"
#include "src/common/system/fdinfo.h"
#include "src/common/system/proc_pid_path.h"
#include "src/common/testing/test_environment.h"
#include "src/common/testing/testing.h"

DECLARE_string(proc_path);

namespace gml::system {

using ::testing::ElementsAre;
using ::testing::MatchesRegex;
using ::testing::UnorderedElementsAre;

constexpr char kTestDataBasePath[] = "src/common/system";

namespace {
std::string GetPathToTestDataFile(std::string_view fname) {
  return bazel::RunfilePath(std::filesystem::path(kTestDataBasePath) / fname);
}
}  // namespace

class ProcParserTest : public ::testing::Test {
 protected:
  void SetUp() override {
    parser_ = std::make_unique<ProcParser>();
    bytes_per_page_ = 4096;
    kernel_tick_time_ns_ = 100;
  }

  std::unique_ptr<ProcParser> parser_;
  int bytes_per_page_ = 0;
  int kernel_tick_time_ns_ = 0;
};

TEST_F(ProcParserTest, ParseNetworkStat) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  std::vector<ProcParser::NetworkStats> stats;
  GML_CHECK_OK(parser_->ParseProcPIDNetDev(123, &stats));

  // The expected values are from the test file above.
  EXPECT_EQ(1, stats.size());
  EXPECT_EQ("ens33", stats[0].interface);
  EXPECT_EQ(54504114, stats[0].rx_bytes);
  EXPECT_EQ(65296, stats[0].rx_packets);
  EXPECT_EQ(0, stats[0].rx_drops);
  EXPECT_EQ(0, stats[0].rx_errs);

  EXPECT_EQ(4258632, stats[0].tx_bytes);
  EXPECT_EQ(39739, stats[0].tx_packets);
  EXPECT_EQ(0, stats[0].tx_drops);
  EXPECT_EQ(0, stats[0].tx_errs);
}

TEST_F(ProcParserTest, ParseStatIO) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  ProcParser::ProcessStats stats;
  GML_CHECK_OK(parser_->ParseProcPIDStatIO(123, &stats));

  // The expected values are from the test file above.
  EXPECT_EQ(5405203, stats.rchar_bytes);
  EXPECT_EQ(1239158, stats.wchar_bytes);
  EXPECT_EQ(17838080, stats.read_bytes);
  EXPECT_EQ(634880, stats.write_bytes);
}

TEST_F(ProcParserTest, ParsePidStat) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  ProcParser::ProcessStats stats;
  GML_CHECK_OK(parser_->ParseProcPIDStat(123, bytes_per_page_, kernel_tick_time_ns_, &stats));

  // The expected values are from the test file above.
  EXPECT_EQ("npm (start)", stats.process_name);

  EXPECT_EQ(800, stats.utime_ns);
  EXPECT_EQ(2300, stats.ktime_ns);
  EXPECT_EQ(13, stats.num_threads);

  EXPECT_EQ(55, stats.major_faults);
  EXPECT_EQ(1799, stats.minor_faults);

  EXPECT_EQ(114384896, stats.vsize_bytes);
  EXPECT_EQ(2577 * bytes_per_page_, stats.rss_bytes);
}

TEST_F(ProcParserTest, ParsePidStatLargePageSize) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  int64_t large_page_size = 2147483648;  // 2.1 GB (INT_MAX + 1)
  ProcParser::ProcessStats stats;
  GML_CHECK_OK(parser_->ParseProcPIDStat(123, large_page_size, kernel_tick_time_ns_, &stats));

  // The expected values are from the test file above.
  EXPECT_EQ("npm (start)", stats.process_name);

  EXPECT_EQ(800, stats.utime_ns);
  EXPECT_EQ(2300, stats.ktime_ns);
  EXPECT_EQ(13, stats.num_threads);

  EXPECT_EQ(55, stats.major_faults);
  EXPECT_EQ(1799, stats.minor_faults);

  EXPECT_EQ(114384896, stats.vsize_bytes);
  EXPECT_EQ(2577 * large_page_size, stats.rss_bytes);
}

TEST_F(ProcParserTest, ParsePSS) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  const size_t pss_bytes = parser_->ParseProcPIDPss(123).ConsumeValueOrDie();
  EXPECT_EQ(pss_bytes, 5936128);
}

TEST_F(ProcParserTest, ParseStat) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  ProcParser::SystemStats stats;
  GML_CHECK_OK(parser_->ParseProcStat(&stats, 2));

  // The expected values are from the test file above.
  EXPECT_EQ(2 * 248758, stats.cpu_utime_ns);
  EXPECT_EQ(2 * 78314, stats.cpu_ktime_ns);
}

TEST_F(ProcParserTest, ParseStatAllCPUs) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  std::vector<ProcParser::CPUStats> stats;
  GML_CHECK_OK(parser_->ParseProcStatAllCPUs(&stats, 2));

  EXPECT_EQ(6, stats.size());

  // The expected values are from the test file above.
  EXPECT_EQ(2 * 41801, stats[2].cpu_utime_ns);
  EXPECT_EQ(2 * 14218, stats[2].cpu_ktime_ns);
  EXPECT_EQ(2 * 2155546, stats[2].cpu_idletime_ns);
  EXPECT_EQ(2 * 1840, stats[2].cpu_iowaittime_ns);
}

TEST_F(ProcParserTest, ParseMemInfo) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  ProcParser::SystemStats stats;
  GML_CHECK_OK(parser_->ParseProcMemInfo(&stats));

  // The expected values are from the test file above.
  EXPECT_EQ(67228110848, stats.mem_total_bytes);
  EXPECT_EQ(17634656256, stats.mem_free_bytes);
  EXPECT_EQ(51960180736, stats.mem_available_bytes);

  EXPECT_EQ(6654636032, stats.mem_buffer_bytes);
  EXPECT_EQ(25549463552, stats.mem_cached_bytes);
  EXPECT_EQ(24576, stats.mem_swap_cached_bytes);

  EXPECT_EQ(28388524032, stats.mem_active_bytes);
  EXPECT_EQ(15734595584, stats.mem_inactive_bytes);
}

TEST_F(ProcParserTest, ParsePIDStatus) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  ProcParser::ProcessStatus stats;
  GML_CHECK_OK(parser_->ParseProcPIDStatus(789, &stats));

  EXPECT_EQ(24612 * 1024, stats.vm_peak_bytes);
  EXPECT_EQ(24612 * 1024, stats.vm_size_bytes);
  EXPECT_EQ(0 * 1024, stats.vm_lck_bytes);
  EXPECT_EQ(0 * 1024, stats.vm_pin_bytes);
  EXPECT_EQ(9936 * 1024, stats.vm_hwm_bytes);
  EXPECT_EQ(9900 * 1024, stats.vm_rss_bytes);
  EXPECT_EQ(3520 * 1024, stats.rss_anon_bytes);
  EXPECT_EQ(6380 * 1024, stats.rss_file_bytes);
  EXPECT_EQ(0 * 1024, stats.rss_shmem_bytes);
  EXPECT_EQ(3640 * 1024, stats.vm_data_bytes);
  EXPECT_EQ(132 * 1024, stats.vm_stk_bytes);
  EXPECT_EQ(2044 * 1024, stats.vm_exe_bytes);
  EXPECT_EQ(5868 * 1024, stats.vm_lib_bytes);
  EXPECT_EQ(76 * 1024, stats.vm_pte_bytes);
  EXPECT_EQ(0 * 1024, stats.vm_swap_bytes);
  EXPECT_EQ(0 * 1024, stats.hugetlb_pages_bytes);

  EXPECT_EQ(2, stats.voluntary_ctxt_switches);
  EXPECT_EQ(2, stats.nonvoluntary_ctxt_switches);
}

TEST_F(ProcParserTest, ParsePIDStatusBadUnit) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  ProcParser::ProcessStatus stats;
  GML_CHECK_OK(parser_->ParseProcPIDStatus(1, &stats));

  // Bad units.
  EXPECT_EQ(-1, stats.vm_peak_bytes);
  // Bad units.
  EXPECT_EQ(-1, stats.vm_size_bytes);
  // Missing key.
  EXPECT_EQ(0, stats.vm_lck_bytes);
  // Exists, and units OK.
  EXPECT_EQ(1 * 1024, stats.vm_pin_bytes);
  // Exists, and no units.
  EXPECT_EQ(123, stats.voluntary_ctxt_switches);
}

TEST_F(ProcParserTest, ParsePIDSMaps) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  std::vector<ProcParser::ProcessSMaps> stats;
  GML_CHECK_OK(parser_->ParseProcPIDSMaps(789, &stats));

  EXPECT_EQ(stats.size(), 5);

  ASSERT_GT(stats.size(), 0);
  auto& first = stats.front();
  EXPECT_EQ("55e816b37000-55e816b65000", first.ToAddress());
  EXPECT_EQ("00000000", first.offset);
  EXPECT_EQ("/usr/bin/vim.basic", first.pathname);
  EXPECT_EQ(184 * 1024, first.size_bytes);
  EXPECT_EQ(4 * 1024, first.kernel_page_size_bytes);
  EXPECT_EQ(4 * 1024, first.mmu_page_size_bytes);
  EXPECT_EQ(184 * 1024, first.rss_bytes);
  EXPECT_EQ(184 * 1024, first.pss_bytes);
  EXPECT_EQ(0 * 1024, first.shared_clean_bytes);
  EXPECT_EQ(0 * 1024, first.shared_dirty_bytes);
  EXPECT_EQ(184 * 1024, first.private_clean_bytes);
  EXPECT_EQ(0 * 1024, first.private_dirty_bytes);
  EXPECT_EQ(184 * 1024, first.referenced_bytes);
  EXPECT_EQ(0 * 1024, first.anonymous_bytes);
  EXPECT_EQ(0 * 1024, first.lazy_free_bytes);
  EXPECT_EQ(0 * 1024, first.anon_huge_pages_bytes);
  EXPECT_EQ(0 * 1024, first.shmem_pmd_mapped_bytes);
  EXPECT_EQ(0 * 1024, first.file_pmd_mapped_bytes);
  EXPECT_EQ(0 * 1024, first.shared_hugetlb_bytes);
  EXPECT_EQ(0 * 1024, first.private_hugetlb_bytes);
  EXPECT_EQ(0 * 1024, first.swap_bytes);
  EXPECT_EQ(0 * 1024, first.swap_pss_bytes);
  EXPECT_EQ(0 * 1024, first.locked_bytes);

  auto& last = stats.back();
  EXPECT_EQ("ffffffffff600000-ffffffffff601000", last.ToAddress());
  EXPECT_EQ("00000000", last.offset);
  EXPECT_EQ("[vsyscall]", last.pathname);
  EXPECT_EQ(4 * 1024, last.size_bytes);
  EXPECT_EQ(4 * 1024, last.kernel_page_size_bytes);
  EXPECT_EQ(4 * 1024, last.mmu_page_size_bytes);
  EXPECT_EQ(0 * 1024, last.rss_bytes);
  EXPECT_EQ(0 * 1024, last.pss_bytes);
  EXPECT_EQ(0 * 1024, last.shared_clean_bytes);
  EXPECT_EQ(0 * 1024, last.shared_dirty_bytes);
  EXPECT_EQ(0 * 1024, last.private_clean_bytes);
  EXPECT_EQ(0 * 1024, last.private_dirty_bytes);
  EXPECT_EQ(0 * 1024, last.referenced_bytes);
  EXPECT_EQ(0 * 1024, last.anonymous_bytes);
  EXPECT_EQ(0 * 1024, last.lazy_free_bytes);
  EXPECT_EQ(0 * 1024, last.anon_huge_pages_bytes);
  EXPECT_EQ(0 * 1024, last.shmem_pmd_mapped_bytes);
  EXPECT_EQ(0 * 1024, last.file_pmd_mapped_bytes);
  EXPECT_EQ(0 * 1024, last.shared_hugetlb_bytes);
  EXPECT_EQ(0 * 1024, last.private_hugetlb_bytes);
  EXPECT_EQ(0 * 1024, last.swap_bytes);
  EXPECT_EQ(0 * 1024, last.swap_pss_bytes);
  EXPECT_EQ(0 * 1024, last.locked_bytes);

  EXPECT_EQ("[anonymous]", stats[3].pathname);
}

TEST_F(ProcParserTest, ReadPIDStartTime) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  ASSERT_OK_AND_EQ(parser_->GetPIDStartTimeTicks(123), 14329);
}

TEST_F(ProcParserTest, ReadPIDCmdline) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  EXPECT_THAT("/usr/lib/slack/slack --force-device-scale-factor=1.5 --high-dpi-support=1",
              parser_->GetPIDCmdline(123));
}

TEST_F(ProcParserTest, ReadPIDMetadataNull) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  EXPECT_THAT("/usr/lib/at-spi2-core/at-spi2-registryd --use-gnome-session",
              parser_->GetPIDCmdline(456));
}

// This test does not work because bazel uses symlinks itself,
// which then causes ReadProcPIDFDLink to resolve the wrong link.
TEST_F(ProcParserTest, ReadProcFDLink) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  {
    // Bazel doesn't copy symlink testdata as symlinks, so we create the missing symlink testdata
    // here.
    ASSERT_OK(
        fs::CreateSymlinkIfNotExists("/dev/null", GetPathToTestDataFile("testdata/proc/123/fd/0")));
    ASSERT_OK(
        fs::CreateSymlinkIfNotExists("/foobar", GetPathToTestDataFile("testdata/proc/123/fd/1")));
    ASSERT_OK(fs::CreateSymlinkIfNotExists("socket:[12345]",
                                           GetPathToTestDataFile("testdata/proc/123/fd/2")));
  }

  std::string out;
  Status s;

  s = parser_->ReadProcPIDFDLink(123, 0, &out);
  EXPECT_OK(s);
  EXPECT_EQ("/dev/null", out);

  s = parser_->ReadProcPIDFDLink(123, 1, &out);
  EXPECT_OK(s);
  EXPECT_EQ("/foobar", out);

  s = parser_->ReadProcPIDFDLink(123, 2, &out);
  EXPECT_OK(s);
  EXPECT_EQ("socket:[12345]", out);

  s = parser_->ReadProcPIDFDLink(123, 3, &out);
  EXPECT_NOT_OK(s);
}

TEST_F(ProcParserTest, ReadUIDs) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  ProcParser::ProcUIDs uids;
  ASSERT_OK(parser_->ReadUIDs(123, &uids));
  EXPECT_EQ("33", uids.real);
  EXPECT_EQ("34", uids.effective);
  EXPECT_EQ("35", uids.saved_set);
  EXPECT_EQ("36", uids.filesystem);
}

TEST_F(ProcParserTest, ReadNSPid) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  std::vector<std::string> ns_pids;
  ASSERT_OK(parser_->ReadNSPid(123, &ns_pids));
  EXPECT_THAT(ns_pids, ElementsAre("2578", "24", "25"));
}

bool operator==(const ProcParser::MountInfo& lhs, const ProcParser::MountInfo& rhs) {
  return lhs.dev == rhs.dev && lhs.root == rhs.root && lhs.mount_point == rhs.mount_point;
}

TEST_F(ProcParserTest, ReadMountInfos) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  {
    std::vector<ProcParser::MountInfo> mount_infos;
    EXPECT_OK(parser_->ReadMountInfos(123, &mount_infos));
    EXPECT_THAT(mount_infos, ElementsAre(
                                 ProcParser::MountInfo{
                                     "260:3",
                                     "/",
                                     "/",
                                 },
                                 ProcParser::MountInfo{"259:3", "/test_foo", "/foo"},
                                 ProcParser::MountInfo{"260:3", "/test_bar", "/bar"}));
  }
  {
    std::vector<ProcParser::MountInfo> mount_infos;
    EXPECT_OK(parser_->ReadMountInfos(1, &mount_infos));
    EXPECT_THAT(mount_infos, ElementsAre(ProcParser::MountInfo{"259:3", "/", "/tmp"}));
  }
}

TEST_F(ProcParserTest, GetMapPaths) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  {
    EXPECT_OK_AND_THAT(
        parser_->GetMapPaths(123),
        UnorderedElementsAre(
            "/dev/zero (deleted)", "/lib/x86_64-linux-gnu/libc-2.28.so",
            "/lib/x86_64-linux-gnu/libdl-2.28.so", "/usr/lib/x86_64-linux-gnu/libcrypto.so.1.1",
            "/usr/lib/x86_64-linux-gnu/libssl.so.1.1", "/usr/sbin/nginx", "/[aio] (deleted)",
            "[heap]", "[stack]", "[uprobes]", "[vdso]", "[vsyscall]", "[vvar]"));
  }
}

TEST_F(ProcParserTest, GetExecutableMapEntry) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));
  {
    ProcParser::ProcessSMaps m;
    m.vmem_start = 0x565078f8c000;
    m.vmem_end = 0x565079054000;
    m.permissions = "r-xp";
    m.pathname = "/usr/sbin/nginx";
    auto smap = parser_->GetExecutableMapEntry(123, "/usr/sbin/nginx", m.vmem_start);
    EXPECT_OK_AND_THAT(smap, m);
  }
}

// Check ProcParser can detect itself.
TEST(ProcParserGetExePathTest, CheckTestProcess) {
  // Since bazel prepares test files as symlinks, creating testdata/proc/123/exe symlink would
  // result into bazel creating another symlink to it. So instead we just use the actual system
  // config to read the actual exe path of this test process.
  ProcParser parser;
  const std::string kExpectedPathRegex = ".*/src/common/system/proc_parser_test";
  ASSERT_OK_AND_ASSIGN(std::filesystem::path proc_exe, parser.GetExePath(getpid()));
  EXPECT_THAT(proc_exe.string(), MatchesRegex(kExpectedPathRegex));
}

TEST_F(ProcParserTest, ParseProcPIDFDInfo) {
  GML_SET_FOR_SCOPE(FLAGS_proc_path, GetPathToTestDataFile("testdata/proc"));

  std::vector<FDInfo> fdinfos;
  ASSERT_OK(parser_->ParseProcPIDFDInfo("1", &fdinfos));
  EXPECT_THAT(
      fdinfos,
      ::testing::UnorderedElementsAre(
          ::testing::AllOf(::testing::Field(&FDInfo::fd, 0), ::testing::Field(&FDInfo::pos, 0),
                           ::testing::Field(&FDInfo::flags, 2004000),
                           ::testing::Field(&FDInfo::mnt_id, 14),
                           ::testing::Field(&FDInfo::inode, 4040260)),
          ::testing::AllOf(
              ::testing::Field(&FDInfo::fd, 1), ::testing::Field(&FDInfo::pos, 0),
              ::testing::Field(&FDInfo::flags, 2100002), ::testing::Field(&FDInfo::mnt_id, 763),
              ::testing::Field(&FDInfo::inode, 53),
              ::testing::Field(
                  &FDInfo::ext,
                  ::testing::Property(
                      &std::unique_ptr<FDInfoExtension>::get,
                      ::testing::WhenDynamicCastTo<DRMFDInfo*>(::testing::AllOf(
                          ::testing::Property(&DRMFDInfo::driver, "i915"),
                          ::testing::Property(&DRMFDInfo::pdev, "0000:00:02.0"),
                          ::testing::Property(&DRMFDInfo::client_id, 116),
                          ::testing::Property(&DRMFDInfo::engines,
                                              ::testing::UnorderedElementsAreArray(
                                                  std::vector<absl::flat_hash_map<
                                                      std::string, DRMFDInfo::Engine>::value_type>{
                                                      {
                                                          "render",
                                                          DRMFDInfo::Engine{
                                                              .busy_ns = 23144220094,
                                                              .capacity = 1,
                                                          },
                                                      },
                                                      {
                                                          "copy",
                                                          DRMFDInfo::Engine{
                                                              .busy_ns = 0,
                                                              .capacity = 1,
                                                          },
                                                      },
                                                      {
                                                          "video",
                                                          DRMFDInfo::Engine{
                                                              .busy_ns = 0,
                                                              .capacity = 2,
                                                          },
                                                      },
                                                      {
                                                          "video-enhance",
                                                          DRMFDInfo::Engine{
                                                              .busy_ns = 0,
                                                              .capacity = 1,
                                                          },
                                                      }}))))))),
          ::testing::AllOf(::testing::Field(&FDInfo::fd, 2), ::testing::Field(&FDInfo::pos, 0),
                           ::testing::Field(&FDInfo::flags, 2004002),
                           ::testing::Field(&FDInfo::mnt_id, 15),
                           ::testing::Field(&FDInfo::inode, 1053))));
}

bool operator==(const DRMFDInfo::Engine& a, const DRMFDInfo::Engine& b) {
  return (a.busy_ns == b.busy_ns) && (a.capacity == b.capacity);
}

}  // namespace gml::system
