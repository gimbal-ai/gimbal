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
use safetensors::{SafeTensors as RustSafeTensors, SafeTensorError, Dtype as RustDtype};

#[cxx::bridge(namespace = "gml::safetensors::rust")]
mod ffi {
    enum Dtype {
        BOOL,
        U8,
        I8,
        F8_E5M2,
        F8_E4M3,
        I16,
        U16,
        F16,
        BF16,
        I32,
        U32,
        F32,
        F64,
        I64,
        U64,
    }

    struct TensorView<'a> {
        dtype: Dtype,
        shape: Vec<usize>,
        data: &'a[u8],
    }

    extern "Rust" {
        type SafeTensors<'a>;

        unsafe fn deserialize<'a>(buffer: &'a[u8]) -> Result<Box<SafeTensors<'a>>>;
        unsafe fn tensor<'a>(safetensors: &'a SafeTensors, name: &str) -> Result<TensorView<'a>>;
        fn names(safetensors: &SafeTensors) -> Vec<String>;
        fn len(safetensors: &SafeTensors) -> usize;
        fn is_empty(safetensors: &SafeTensors) -> bool;
    }
}

pub struct SafeTensors<'a> {
    inner: safetensors::SafeTensors<'a>,
}

pub fn deserialize<'a>(buffer: &'a[u8]) -> Result<Box<SafeTensors<'a>>, safetensors::SafeTensorError> {
    // Deserialize the SafeTensors using the memory-mapped file
    Ok(Box::new(SafeTensors { inner: RustSafeTensors::deserialize(buffer)? }))
}

pub unsafe fn tensor<'a>(safetensors: &'a SafeTensors, name: &str) -> Result<ffi::TensorView<'a>, SafeTensorError> {
    let tensor_view = safetensors.inner.tensor(name)?;
    let dtype = match tensor_view.dtype() {
        RustDtype::BOOL => ffi::Dtype::BOOL,
        RustDtype::U8 => ffi::Dtype::U8,
        RustDtype::I8 => ffi::Dtype::I8,
        RustDtype::F8_E5M2 => ffi::Dtype::F8_E5M2,
        RustDtype::F8_E4M3 => ffi::Dtype::F8_E4M3,
        RustDtype::I16 => ffi::Dtype::I16,
        RustDtype::U16 => ffi::Dtype::U16,
        RustDtype::F16 => ffi::Dtype::F16,
        RustDtype::BF16 => ffi::Dtype::BF16,
        RustDtype::I32 => ffi::Dtype::I32,
        RustDtype::U32 => ffi::Dtype::U32,
        RustDtype::F32 => ffi::Dtype::F32,
        RustDtype::F64 => ffi::Dtype::F64,
        RustDtype::I64 => ffi::Dtype::I64,
        RustDtype::U64 => ffi::Dtype::U64,
        _ => todo!(),
    };
    Ok(ffi::TensorView {
        dtype,
        shape: tensor_view.shape().to_vec(),
        data: tensor_view.data(),
    })
}

pub fn names(safetensors: &SafeTensors) -> Vec<String> {
    safetensors.inner.names().iter().map(|s| s.to_string()).collect()
}

pub fn len(safetensors: &SafeTensors) -> usize {
    safetensors.inner.len()
}

pub fn is_empty(safetensors: &SafeTensors) -> bool {
    safetensors.inner.is_empty()
}
