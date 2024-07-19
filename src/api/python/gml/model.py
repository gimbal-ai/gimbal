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
import io
from pathlib import Path
from typing import BinaryIO, Dict, List, Literal, Optional, TextIO, Tuple

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb
import torch
import torch_mlir
from gml.compile import to_torch_mlir
from gml.preprocessing import ImagePreprocessingStep
from gml.tensor import TensorSemantics


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

    def to_proto(self) -> modelexecpb.ModelInfo:
        image_preprocessing_steps = None
        if self.image_preprocessing_steps:
            image_preprocessing_steps = [
                step.to_proto() for step in self.image_preprocessing_steps
            ]
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
        )

    @abc.abstractmethod
    def collect_assets(self) -> Dict[str, TextIO | BinaryIO | Path]:
        pass


class TorchModel(Model):
    def __init__(
        self,
        name: str,
        torch_module: torch.nn.Module,
        input_shapes: List[List[int]],
        input_dtypes: List[torch.dtype],
        **kwargs,
    ):
        super().__init__(
            name,
            modelexecpb.ModelInfo.MODEL_KIND_TORCHSCRIPT,
            modelexecpb.ModelInfo.MODEL_STORAGE_FORMAT_MLIR_TEXT,
            **kwargs,
        )
        self.torch_module = torch_module
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes

    def _convert_to_torch_mlir(self):
        return to_torch_mlir(
            self.torch_module.to("cpu"),
            [
                torch_mlir.TensorPlaceholder(shape, dtype)
                for shape, dtype in zip(self.input_shapes, self.input_dtypes)
            ],
        )

    def collect_assets(self) -> Dict[str, TextIO | BinaryIO | Path]:
        compiled = self._convert_to_torch_mlir()
        file = io.BytesIO(str(compiled).encode("utf-8"))
        return {"": file}


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

    def collect_assets(self) -> Dict[str, TextIO | BinaryIO | Path]:
        return self.files
