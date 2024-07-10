# Copyright 2023- Gimlet Labs, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import sys

import gml
import gml.compile
import pytest
import torch
from torch_mlir import TensorPlaceholder


class AddModule(torch.nn.Module):
    def forward(self, x):
        return x + x


def test_add_module_to_torch_mlir():
    model = AddModule()
    example_inputs = [TensorPlaceholder([1, 2, 3], dtype=torch.float32)]

    compiled = gml.compile.to_torch_mlir(model, example_inputs)

    op_names = []
    for op in compiled.body.operations[0].body.blocks[0].operations:
        op_names.append(op.operation.name)

    assert "torch.aten.add.Tensor" in op_names


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
