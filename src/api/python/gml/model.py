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


def box_format_str_to_proto(box_format: str):
    match box_format:
        case "cxcywh":
            return modelexecpb.BoundingBoxInfo.BOUNDING_BOX_FORMAT_CXCYWH
        case "xyxy":
            return modelexecpb.BoundingBoxInfo.BOUNDING_BOX_FORMAT_XYXY
        case "yxyx":
            return modelexecpb.BoundingBoxInfo.BOUNDING_BOX_FORMAT_YXYX
        case _:
            raise ValueError("Invalid bounding box format: {}".format(box_format))


class BoundingBoxFormat:
    def __init__(self, box_format: str = "cxcywh", is_normalized: bool = True):
        self.box_format = box_format_str_to_proto(box_format)
        self.is_normalized = is_normalized

    def to_proto(self) -> modelexecpb.BoundingBoxInfo:
        return modelexecpb.BoundingBoxInfo(
            box_format=self.box_format,
            box_normalized=self.is_normalized,
        )


class Model:
    def __init__(
        self,
        name: str,
        torch_module: torch.nn.Module,
        input_shapes: List[List[int]],
        input_dtypes: List[torch.dtype],
        output_bbox_format: Optional[BoundingBoxFormat] = None,
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
        self.output_bbox_format = output_bbox_format
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
        bbox_info = None
        if self.output_bbox_format:
            bbox_info = self.output_bbox_format.to_proto()
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
            bbox_info=bbox_info,
            image_preprocessing_steps=image_preprocessing_steps,
        )
