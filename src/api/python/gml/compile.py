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

import contextlib
import functools
from typing import Dict, List, Optional, Sequence, Union

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb
import torch
import torch_mlir  # noqa

try:
    import torch_mlir.fx  # noqa
    from torch.export import export  # noqa

    has_fx_importer_torch_export = True
except ImportError:
    has_fx_importer_torch_export = False


def _default_decomposition_denylist():
    """These ops will not be decomposed by default."""
    return [
        torch.ops.aten.full.default,
        torch.ops.aten.upsample_bilinear2d.vec,
    ]


@contextlib.contextmanager
def _patch_aot_export_module():
    """This contextmanager prevents PyTorch dispatch from running when calling aot_export_module.

    This patch is necessary because not all callers of `aot_export_module` expose the pre_dispatch flag.
    For example, `ExportedProgram.run_decompositions` which is called by `torch_mlir.fx.export_and_import` doesn't
    expose the pre_dispatch flag.

    Without setting `pre_dispatch=True`, PyTorch dispatch will run before tracing which causes certain operations to be decomposed.
    For example, `upsample_nearest2d` will be decomposed into aten.index.Tensor calls. This is undesirable for runtimes that provide
    optimized implementations of the equivalent of `upsample_nearest2d`.
    """
    import torch._functorch.aot_autograd

    orig = torch._functorch.aot_autograd.aot_export_module
    torch._functorch.aot_autograd.aot_export_module = functools.partial(
        orig, pre_dispatch=True
    )
    yield
    torch._functorch.aot_autograd.aot_export_module = orig


def to_torch_mlir_w_torch_export(
    model: torch.nn.Module,
    example_inputs: Sequence[torch.Tensor],
    dynamic_shapes: Optional[
        Sequence[Dict[int, Union[str, "torch.export.dynamic_shapes._Dim"]]]
    ] = None,
    decomposition_denylist: Optional[List[torch._ops.OperatorBase]] = None,
):
    from torch._decomp import remove_decompositions
    from torch.export._trace import _export
    from torch_mlir.extras.fx_decomp_util import get_decomposition_table
    from torch_mlir.fx import export_and_import

    if dynamic_shapes is not None:
        for shape in dynamic_shapes:
            if not isinstance(shape, dict):
                continue
            for idx in shape:
                if isinstance(shape[idx], torch.export.dynamic_shapes._Dim):
                    continue
                shape[idx] = torch.export.Dim(shape[idx])

    if decomposition_denylist is None:
        decomposition_denylist = _default_decomposition_denylist()

    model = model.eval().to("cpu")

    try:
        # Running the model a few times on the inputs, leads to more consistent compiled results.
        for _ in range(2):
            _ = model(*example_inputs)
    except:  # noqa
        # Ignore errors running the model. This can happen when the model has data dependent branches.
        pass

    prog = _export(
        model,
        tuple(example_inputs),
        pre_dispatch=False,
        strict=False,
        dynamic_shapes=dynamic_shapes,
    )
    decomp_table = get_decomposition_table()
    remove_decompositions(decomp_table, decomposition_denylist)
    with _patch_aot_export_module():
        return export_and_import(
            prog,
            *example_inputs,
            decomposition_table=decomp_table,
        )


def to_torch_mlir_fallback(model, example_inputs):
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch_mlir import ExampleArgs, OutputType
    from torch_mlir import compile as torch_mlir_compile
    from torch_mlir.dynamo import _get_decomposition_table

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


def to_torch_mlir(
    model,
    example_inputs,
    dynamic_shapes: Optional[
        Sequence[Dict[int, Union[str, "torch.export.dynamic_shapes._Dim"]]]
    ] = None,
):
    if has_fx_importer_torch_export:
        return to_torch_mlir_w_torch_export(model, example_inputs, dynamic_shapes)
    else:
        return to_torch_mlir_fallback(model, example_inputs)


def torch_mlir_output_kind():
    if has_fx_importer_torch_export:
        return modelexecpb.ModelInfo.MODEL_KIND_TORCH
    else:
        return modelexecpb.ModelInfo.MODEL_KIND_TORCHSCRIPT
