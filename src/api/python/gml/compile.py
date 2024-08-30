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
from typing import Any, Dict, List, Optional, Sequence, Union

import safetensors_mlir
import torch
import torch_mlir
from gml.asset_manager import AssetManager
from mlir.ir import (
    BF16Type,
    ComplexType,
    Context,
    F16Type,
    F32Type,
    F64Type,
    IntegerType,
    Operation,
    RankedTensorType,
    Value,
)
from safetensors.torch import save_file
from torch._decomp import remove_decompositions
from torch.export._trace import _export
from torch_mlir.dialects import torch as torch_d
from torch_mlir.extras.fx_decomp_util import get_decomposition_table
from torch_mlir.extras.fx_importer import FxImporter, FxImporterHooks
from torch_mlir.fx import export_and_import


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


_torch_dtype_to_builtin_element_type = {
    torch.float16: lambda: F16Type.get(),
    torch.bfloat16: lambda: BF16Type.get(),
    torch.float32: lambda: F32Type.get(),
    torch.float64: lambda: F64Type.get(),
    torch.uint8: lambda: IntegerType.get_unsigned(8),
    torch.int8: lambda: IntegerType.get_signless(8),
    torch.int16: lambda: IntegerType.get_signless(16),
    torch.int32: lambda: IntegerType.get_signless(32),
    torch.int64: lambda: IntegerType.get_signless(64),
    torch.bool: lambda: IntegerType.get_signless(1),
    torch.qint8: lambda: IntegerType.get_signless(8),
    torch.quint8: lambda: IntegerType.get_unsigned(8),
    torch.complex32: lambda: ComplexType.get(F16Type.get()),
    torch.complex64: lambda: ComplexType.get(F32Type.get()),
    torch.complex128: lambda: ComplexType.get(F64Type.get()),
}


def _get_unique_(tensors, name):
    index = 0
    name = "{}_{}".format(name, index)
    while name in tensors:
        index += 1
        name = "{}_{}".format(name, index)
    return name


class TensorSet:
    def __init__(self):
        self._tensors: Dict[str, torch.Tensor] = dict()

    def add(self, tensor: torch.Tensor) -> str:
        shape_desc = "_".join([str(d) for d in tensor.shape])
        base_name = f"torch_tensor_{shape_desc}_{str(tensor.dtype)}"

        index = 0
        name = "{}_{}".format(base_name, index)
        while name in self._tensors and not torch.equal(tensor, self._tensors[name]):
            index += 1
            name = "{}_{}".format(base_name, index)

        self._tensors[name] = tensor
        return name

    def tensors(self) -> Dict[str, torch.Tensor]:
        return self._tensors


class SafetensorImporterHooks(FxImporterHooks):
    def __init__(self, asset_manager: AssetManager):
        self._asset_mgr = asset_manager
        # TODO(james): shard weights into multiple shards.
        self.asset_name = "weights.shard0"
        self._tensors = TensorSet()

    def resolve_literal(
        self, gni: "torch_mlir.extras.fx_importer.GraphNodeImporter", literal: Any
    ) -> Optional[Value]:
        if not isinstance(literal, torch.Tensor):
            return None
        tensor = literal
        ctx = gni._c

        tensor_name = self._tensors.add(tensor)

        file_attr = safetensors_mlir.FileAttr.get(ctx, self.asset_name)

        if tensor.dtype not in _torch_dtype_to_builtin_element_type:
            raise ValueError("unsupported torch dtype: {}".format(tensor.dtype))
        elem_type = _torch_dtype_to_builtin_element_type[tensor.dtype]()
        tensor_type = RankedTensorType.get(tuple(tensor.size()), elem_type)

        tensor_attr = safetensors_mlir.TensorAttr.get(
            tensor_type, file_attr, tensor_name
        )
        builtin_tensor = safetensors_mlir.tensor_ref(tensor_type, tensor_attr)

        vtensor_type = gni._cc.tensor_to_vtensor_type(tensor)
        return Operation.create(
            name="torch_c.from_builtin_tensor",
            results=[vtensor_type],
            operands=[builtin_tensor],
        ).result

    def save_tensors(self):
        file_path = self._asset_mgr.add_asset(self.asset_name)
        tensors = self._tensors.tensors()
        for k in tensors:
            tensors[k] = tensors[k].contiguous()
        save_file(tensors, file_path)


def to_torch_mlir(
    model: torch.nn.Module,
    example_inputs: Sequence[torch.Tensor],
    dynamic_shapes: Optional[
        Sequence[Dict[int, Union[str, "torch.export.dynamic_shapes._Dim"]]]
    ] = None,
    decomposition_denylist: Optional[List[torch._ops.OperatorBase]] = None,
    weight_manager: Optional[AssetManager] = None,
):
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
    hooks = None
    if weight_manager is not None:
        hooks = SafetensorImporterHooks(weight_manager)

    context = Context()
    torch_d.register_dialect(context)
    safetensors_mlir.register_dialect(context)
    fx_importer = FxImporter(context=context, hooks=hooks)

    with _patch_aot_export_module():
        module = export_and_import(
            prog,
            *example_inputs,
            decomposition_table=decomp_table,
            fx_importer=fx_importer,
        )

    if hooks is not None:
        hooks.save_tensors()

    try:
        module.operation.verify()
    except Exception as exc:
        raise Exception(
            "failed to verify converted torch model MLIR module: {}".format(module)
        ) from exc

    return module
