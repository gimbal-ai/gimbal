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
