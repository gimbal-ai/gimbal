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

#include "src/gem/exec/plugin/huggingface/tokenizer_wrapper/hf_fast_tokenizer.h"

#include "src/common/bazel/runfiles.h"
#include "src/common/testing/testing.h"

namespace gml::gem::exec::test {

class HFFastTokenizerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto tok_json = bazel::RunfilePath(
        "src/gem/exec/plugin/huggingface/tokenizer_wrapper/testdata/sample_tokenizer.json");
    tokenizer = HFFastTokenizer::Create(tok_json);
  }

  std::unique_ptr<gml::gem::exec::core::Tokenizer> tokenizer;
};

TEST_F(HFFastTokenizerTest, EncodeDecodeTest) {
  std::string original_text = "This is a test sentence.";
  std::vector<int> encoded_tokens = tokenizer->Encode(original_text);
  std::string decoded_text = tokenizer->Decode(encoded_tokens);

  // Check if the decoded text matches the original text
  EXPECT_EQ(original_text, decoded_text);
}

}  // namespace gml::gem::exec::test
