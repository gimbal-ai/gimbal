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

use cxx::{CxxString, CxxVector};

#[cxx::bridge(namespace = gml::gem::exec)]
mod ffi {
    extern "Rust" {
        type TokenizerWrapper;

        fn new_tokenizer(path: &CxxString) -> Box<TokenizerWrapper>;
        fn encode(self: &TokenizerWrapper, text: &CxxString, add_special_tokens: bool) -> Vec<i32>;
        fn decode(self: &TokenizerWrapper, ids: &CxxVector<i32>, skip_special_tokens: bool) -> String;
    }
}

pub struct TokenizerWrapper {
    tokenizer: tokenizers::Tokenizer,
}

pub fn new_tokenizer(path: &CxxString) -> Box<TokenizerWrapper> {
    let tokenizer = tokenizers::Tokenizer::from_file(path.to_string()).unwrap();
    Box::new(TokenizerWrapper { tokenizer })
}

impl TokenizerWrapper {


    pub fn encode(&self, text: &CxxString, add_special_tokens: bool) -> Vec<i32> {
        let encoding = self.tokenizer.encode(text.to_string(), add_special_tokens).unwrap();
        encoding.get_ids().iter().map(|&id| id as i32).collect()
    }

    pub fn decode(&self, ids: &CxxVector<i32>, skip_special_tokens: bool) -> String {
      let ids_vec: Vec<u32> = ids.iter().map(|&id| id as u32).collect();
      let tokens = self.tokenizer.decode(&ids_vec, skip_special_tokens).unwrap();
      tokens
    }
}
