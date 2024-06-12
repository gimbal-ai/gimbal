# Copyright © 2023- Gimlet Labs, Inc.
# All Rights Reserved.
#
# NOTICE:  All information contained herein is, and remains
# the property of Gimlet Labs, Inc. and its suppliers,
# if any.  The intellectual and technical concepts contained
# herein are proprietary to Gimlet Labs, Inc. and its suppliers and
# may be covered by U.S. and Foreign Patents, patents in process,
# and are protected by trade secret or copyright law. Dissemination
# of this information or reproduction of this material is strictly
# forbidden unless prior written permission is obtained from
# Gimlet Labs, Inc.
#
# SPDX-License-Identifier: Proprietary

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
