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
from __future__ import annotations

import abc
import contextlib
import io
from pathlib import Path
from typing import BinaryIO, Dict, List, Literal, Optional, Sequence, TextIO, Tuple

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb
import torch
from gml.asset_manager import AssetManager, TempFileAssetManager
from gml.compile import to_torch_mlir
from gml.preprocessing import ImagePreprocessingStep
from gml.tensor import TensorSemantics


class GenerationConfig:
    def __init__(self, eos_token_ids: List[int]):
        self.eos_token_ids = eos_token_ids

    def to_proto(self) -> modelexecpb.GenerationConfig:
        return modelexecpb.GenerationConfig(
            eos_token_ids=self.eos_token_ids,
        )


class Model(abc.ABC):
    def __init__(
        self,
        name: str,
        kind: modelexecpb.ModelInfo.ModelKind,
        storage_format: modelexecpb.ModelInfo.ModelStorageFormat,
        input_tensor_semantics: List[TensorSemantics],
        output_tensor_semantics: List[TensorSemantics],
        class_labels: Optional[List[str]] = None,
        class_labels_file: Optional[Path] = None,
        image_preprocessing_steps: Optional[List[ImagePreprocessingStep]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        self.name = name
        self.kind = kind
        self.storage_format = storage_format
        self.class_labels = class_labels
        if class_labels_file:
            self.class_labels = []
            with open(class_labels_file, "r") as f:
                for line in f.readlines():
                    self.class_labels.append(line.strip())
        self.input_tensor_semantics = input_tensor_semantics
        self.output_tensor_semantics = output_tensor_semantics
        self.image_preprocessing_steps = image_preprocessing_steps
        self.generation_config = generation_config

    def to_proto(self) -> modelexecpb.ModelInfo:
        image_preprocessing_steps = None
        if self.image_preprocessing_steps:
            image_preprocessing_steps = [
                step.to_proto() for step in self.image_preprocessing_steps
            ]
        generation_config = None
        if self.generation_config:
            generation_config = self.generation_config.to_proto()
        return modelexecpb.ModelInfo(
            name=self.name,
            kind=self.kind,
            format=self.storage_format,
            class_labels=self.class_labels,
            image_preprocessing_steps=image_preprocessing_steps,
            input_tensor_semantics=[
                semantics.to_proto() for semantics in self.input_tensor_semantics
            ],
            output_tensor_semantics=[
                semantics.to_proto() for semantics in self.output_tensor_semantics
            ],
            generation_config=generation_config,
        )

    @abc.abstractmethod
    def _collect_assets(
        self, weight_manager: Optional[AssetManager] = None
    ) -> Dict[str, TextIO | BinaryIO | Path]:
        pass

    @contextlib.contextmanager
    def collect_assets(self, weight_manager: Optional[AssetManager] = None):
        yield from self._collect_assets(weight_manager)


class TorchModel(Model):
    def __init__(
        self,
        name: str,
        torch_module: torch.nn.Module,
        example_inputs: Optional[List[torch.Tensor]] = None,
        input_shapes: Optional[List[List[int]]] = None,
        input_dtypes: Optional[List[torch.dtype]] = None,
        dynamic_shapes: Optional[Sequence[Dict[int, str | "torch.export.Dim"]]] = None,
        **kwargs,
    ):
        super().__init__(
            name,
            modelexecpb.ModelInfo.MODEL_KIND_TORCH,
            modelexecpb.ModelInfo.MODEL_STORAGE_FORMAT_MLIR_TEXT,
            **kwargs,
        )
        self.torch_module = torch_module
        self.example_inputs = example_inputs
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes
        self.dynamic_shapes = dynamic_shapes
        if self.example_inputs is None:
            if self.input_shapes is None or self.input_dtypes is None:
                raise ValueError(
                    "one of `example_inputs` or (`input_shapes` and `input_dtype`) must be provided to `TorchModel`"
                )
            self.example_inputs = [
                torch.rand(shape, dtype=dtype)
                for shape, dtype in zip(self.input_shapes, self.input_dtypes)
            ]

    def _convert_to_torch_mlir(self, weight_manager: Optional[AssetManager] = None):
        return to_torch_mlir(
            self.torch_module,
            self.example_inputs,
            self.dynamic_shapes,
            weight_manager=weight_manager,
        )

    def _collect_assets(
        self, weight_manager: Optional[AssetManager] = None
    ) -> Dict[str, TextIO | BinaryIO | Path]:
        if weight_manager is None:
            # If the user does not provide a weight manager, use temp files.
            weight_manager = TempFileAssetManager()

        with weight_manager as weight_mgr:
            compiled = self._convert_to_torch_mlir(weight_mgr)
            file = io.BytesIO(str(compiled).encode("utf-8"))
            assets = {"": file}
            assets.update(weight_mgr.assets())
            yield assets


def _kind_str_to_kind_format_protos(
    kind: str,
) -> Tuple[modelexecpb.ModelInfo.ModelKind, modelexecpb.ModelInfo.ModelStorageFormat]:
    match kind.lower():
        case "openvino":
            return (
                modelexecpb.ModelInfo.MODEL_KIND_OPENVINO,
                modelexecpb.ModelInfo.MODEL_STORAGE_FORMAT_OPENVINO,
            )
        case "onnx":
            return (
                modelexecpb.ModelInfo.MODEL_KIND_ONNX,
                modelexecpb.ModelInfo.MODEL_STORAGE_FORMAT_PROTOBUF,
            )
        case "tfl":
            return (
                modelexecpb.ModelInfo.MODEL_KIND_TFLITE,
                modelexecpb.ModelInfo.MODEL_STORAGE_FORMAT_FLATBUFFER,
            )
        case _:
            raise ValueError("invalid model kind: {}".format(kind))


class ModelFromFiles(Model):
    def __init__(
        self,
        name: str,
        kind: Literal["openvino", "onnx", "tfl"],
        files: Dict[str, TextIO | BinaryIO | Path],
        **kwargs,
    ):
        kind, storage_format = _kind_str_to_kind_format_protos(kind)
        super().__init__(name=name, kind=kind, storage_format=storage_format, **kwargs)
        self.files = files

    def _collect_assets(
        self, weight_manager: Optional[AssetManager] = None
    ) -> Dict[str, TextIO | BinaryIO | Path]:
        yield self.files
