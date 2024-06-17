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

from pathlib import Path
from typing import List, Optional

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb
import torch
import torch_mlir
from gml.compile import to_torch_mlir
from gml.preprocessing import ImagePreprocessingStep
from gml.tensor import TensorSemantics


class Model:
    def __init__(
        self,
        name: str,
        torch_module: torch.nn.Module,
        input_shapes: List[List[int]],
        input_dtypes: List[torch.dtype],
        input_tensor_semantics: List[TensorSemantics],
        output_tensor_semantics: List[TensorSemantics],
        class_labels: Optional[List[str]] = None,
        class_labels_file: Optional[Path] = None,
        image_preprocessing_steps: Optional[List[ImagePreprocessingStep]] = None,
    ):
        self.name = name
        self.torch_module = torch_module
        self.class_labels = class_labels
        if class_labels_file:
            self.class_labels = []
            with open(class_labels_file, "r") as f:
                for line in f.readlines():
                    self.class_labels.append(line.strip())
        self.input_tensor_semantics = input_tensor_semantics
        self.output_tensor_semantics = output_tensor_semantics
        self.image_preprocessing_steps = image_preprocessing_steps
        self.input_shapes = input_shapes
        self.input_dtypes = input_dtypes

    def convert_to_torch_mlir(self):
        return to_torch_mlir(
            self.torch_module,
            [
                torch_mlir.TensorPlaceholder(shape, dtype)
                for shape, dtype in zip(self.input_shapes, self.input_dtypes)
            ],
        )

    def to_proto(self) -> modelexecpb.ModelInfo:
        image_preprocessing_steps = None
        if self.image_preprocessing_steps:
            image_preprocessing_steps = [
                step.to_proto() for step in self.image_preprocessing_steps
            ]
        return modelexecpb.ModelInfo(
            name=self.name,
            kind=modelexecpb.ModelInfo.MODEL_KIND_TORCHSCRIPT,
            format=modelexecpb.ModelInfo.MODEL_STORAGE_FORMAT_MLIR_TEXT,
            class_labels=self.class_labels,
            image_preprocessing_steps=image_preprocessing_steps,
            input_tensor_semantics=[
                semantics.to_proto() for semantics in self.input_tensor_semantics
            ],
            output_tensor_semantics=[
                semantics.to_proto() for semantics in self.output_tensor_semantics
            ],
        )
