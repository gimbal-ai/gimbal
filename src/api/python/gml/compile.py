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

from torch.fx.experimental.proxy_tensor import make_fx
from torch_mlir import ExampleArgs, OutputType
from torch_mlir import compile as torch_mlir_compile
from torch_mlir.dynamo import _get_decomposition_table


def to_torch_mlir(model, example_inputs):
    example_args = ExampleArgs.get(example_inputs)
    args = example_args._get_for_tracing(use_tracing=True, ignore_traced_shapes=True)[
        "forward"
    ]
    try:
        # Running the model a few times on the inputs, leads to more consistent compiled results.
        for _ in range(2):
            _ = model(*args)
    except Exception:
        # Ignore errors running the model. This can happen when the model has data dependent branches.
        pass

    try:
        compiled = torch_mlir_compile(
            model,
            example_inputs,
            use_tracing=False,
            ignore_traced_shapes=False,
            output_type=OutputType.RAW,
            use_make_fx=False,
        )
        return compiled
    except Exception:
        pass

    # If the module can't be exported directly, we try to create an FX graph and then export it.
    model = make_fx(
        model, pre_dispatch=True, decomposition_table=_get_decomposition_table()
    )(*args)
    compiled = torch_mlir_compile(
        model,
        example_inputs,
        use_tracing=False,
        ignore_traced_shapes=False,
        output_type=OutputType.RAW,
        use_make_fx=False,
    )

    return compiled
