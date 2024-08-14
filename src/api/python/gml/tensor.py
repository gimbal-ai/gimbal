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

import abc
from typing import List, Literal, Optional, Tuple

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb
import google.protobuf.wrappers_pb2 as wrapperspb
import numpy as np


def box_format_str_to_proto(box_format: str):
    match box_format.lower():
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


class DimensionSemantics(abc.ABC):
    @abc.abstractmethod
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        pass


class BatchDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_BATCH,
        )


class IgnoreDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_IGNORE,
        )


def chan_format_str_to_proto(chan_format: str):
    match chan_format.lower():
        case "rgb":
            return (
                modelexecpb.DimensionSemantics.ImageChannelParams.IMAGE_CHANNEL_FORMAT_RGB
            )
        case "bgr":
            return (
                modelexecpb.DimensionSemantics.ImageChannelParams.IMAGE_CHANNEL_FORMAT_BGR
            )
        case _:
            raise ValueError("Invalid channel_format format: {}".format(chan_format))


class ImageChannelDimension(DimensionSemantics):
    def __init__(self, channel_format="rgb"):
        self.channel_format = chan_format_str_to_proto(channel_format)

    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_IMAGE_CHANNEL,
            image_channel_params=modelexecpb.DimensionSemantics.ImageChannelParams(
                format=self.channel_format,
            ),
        )


class ImageHeightDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_IMAGE_HEIGHT,
        )


class ImageWidthDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_IMAGE_WIDTH,
        )


class DetectionNumCandidatesDimension(DimensionSemantics):
    def __init__(self, is_nms: bool = False):
        self.is_nms = is_nms

    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_DETECTION_CANDIDATES,
            detection_candidates_params=modelexecpb.DimensionSemantics.DetectionCandidatesParams(
                is_nms_boxes=self.is_nms
            ),
        )


class DetectionOutputDimension(DimensionSemantics):
    def __init__(
        self,
        coordinates_start_index: Optional[int] = None,
        box_format: Optional[BoundingBoxFormat] = None,
        box_confidence_index: Optional[int] = None,
        class_index: Optional[int] = None,
        scores_range: Optional[Tuple[int, int]] = None,
        scores_are_logits: bool = False,
    ):
        self.coordinates_range = None
        if coordinates_start_index is not None:
            self.coordinates_range = (coordinates_start_index, 4)
        self.box_format = box_format
        self.box_confidence_index = box_confidence_index
        self.class_index = class_index
        self.scores_range = scores_range
        self.scores_are_logits = scores_are_logits

    def to_proto(self) -> modelexecpb.DimensionSemantics:
        scores_range = None
        if self.scores_range is not None:
            scores_range = (
                modelexecpb.DimensionSemantics.DetectionOutputParams.IndexRange(
                    start=self.scores_range[0],
                    size=self.scores_range[1],
                )
            )
        box_confidence_index = np.iinfo(np.int32).min
        if self.box_confidence_index is not None:
            box_confidence_index = self.box_confidence_index
        box_format_proto = None
        if self.box_format is not None:
            box_format_proto = self.box_format.to_proto()
        box_coordinate_range_proto = None
        if self.coordinates_range is not None:
            box_coordinate_range_proto = (
                modelexecpb.DimensionSemantics.DetectionOutputParams.IndexRange(
                    start=self.coordinates_range[0],
                    size=self.coordinates_range[1],
                )
            )
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_DETECTION_OUTPUT,
            detection_output_params=modelexecpb.DimensionSemantics.DetectionOutputParams(
                box_coordinate_range=box_coordinate_range_proto,
                box_format=box_format_proto,
                box_confidence_index=box_confidence_index,
                class_index=self.class_index,
                scores_range=scores_range,
                scores_are_logits=self.scores_are_logits,
            ),
        )


def _segmentation_mask_kind_to_proto(kind: str):
    match kind.lower():
        case "bool_masks":
            return (
                modelexecpb.DimensionSemantics.SegmentationMaskParams.SEGMENTATION_MASK_KIND_BOOL
            )
        case "int_label_masks":
            return (
                modelexecpb.DimensionSemantics.SegmentationMaskParams.SEGMENTATION_MASK_KIND_CLASS_LABEL
            )
        case "score_mask":
            return (
                modelexecpb.DimensionSemantics.SegmentationMaskParams.SEGMENTATION_MASK_KIND_SCORE
            )
        case "logits_mask":
            return (
                modelexecpb.DimensionSemantics.SegmentationMaskParams.SEGMENTATION_MASK_KIND_LOGITS
            )
        case _:
            raise ValueError("Invalid segmentation mask kind: {}".format(kind))


class SegmentationMaskChannel(DimensionSemantics):
    def __init__(
        self,
        kind: Literal["bool_masks", "int_label_masks", "score_mask", "logits_mask"],
    ):
        self.kind = _segmentation_mask_kind_to_proto(kind)

    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_SEGMENTATION_MASK_CHANNEL,
            segmentation_mask_params=modelexecpb.DimensionSemantics.SegmentationMaskParams(
                kind=self.kind,
            ),
        )


class LabelsDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_CLASS_LABELS
        )


class ScoresDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_CLASS_SCORES
        )


class RegressionValueDimension(DimensionSemantics):
    def __init__(self, label: str, scale: Optional[float] = None):
        self.label = label
        self.scale = scale

    def to_proto(self) -> modelexecpb.DimensionSemantics:
        scale = None
        if self.scale is not None:
            scale = wrapperspb.DoubleValue(value=self.scale)
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_REGRESSION_VALUE,
            regression_params=modelexecpb.DimensionSemantics.RegressionParams(
                label=self.label,
                scale=scale,
            ),
        )


class TokensDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_TOKENS,
        )


class AttentionMaskDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_ATTENTION_MASK,
        )


class VocabLogitsDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_VOCAB_LOGITS,
        )


class EmbeddingDimension(DimensionSemantics):
    def to_proto(self) -> modelexecpb.DimensionSemantics:
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_EMBEDDING,
        )


class TensorSemantics:
    def __init__(
        self,
        dimensions: List[DimensionSemantics],
        kind: modelexecpb.TensorSemantics.TensorSemanticsKind = modelexecpb.TensorSemantics.TENSOR_SEMANTICS_KIND_DIMENSION,
    ):
        self.dimensions = dimensions
        self.kind = kind

    def to_proto(self) -> modelexecpb.TensorSemantics:
        return modelexecpb.TensorSemantics(
            dimensions=[dim.to_proto() for dim in self.dimensions],
            kind=self.kind,
        )


class UnusedTensorSemantics(TensorSemantics):
    def __init__(self):
        super().__init__(
            dimensions=[],
            kind=modelexecpb.TensorSemantics.TENSOR_SEMANTICS_KIND_UNUSED,
        )


class AttentionKeyValueCacheTensorSemantics(TensorSemantics):
    def __init__(self):
        super().__init__(
            dimensions=[],
            kind=modelexecpb.TensorSemantics.TENSOR_SEMANTICS_KIND_ATTENTION_KEY_VALUE_CACHE,
        )


class RGBImage(TensorSemantics):
    """RGBImage is an image tensor input with channels in RGB order."""

    def __init__(self, channels_first=True):
        if channels_first:
            dimensions = [
                BatchDimension(),
                ImageChannelDimension(channel_format="rgb"),
                ImageHeightDimension(),
                ImageWidthDimension(),
            ]
        else:
            dimensions = [
                BatchDimension(),
                ImageHeightDimension(),
                ImageWidthDimension(),
                ImageChannelDimension(channel_format="rgb"),
            ]
        super().__init__(dimensions)


class BGRImage(TensorSemantics):
    """BGRImage is an image tensor input with channels in BGR order."""

    def __init__(self, channels_first=True):
        if channels_first:
            dimensions = [
                BatchDimension(),
                ImageChannelDimension(channel_format="bgr"),
                ImageHeightDimension(),
                ImageWidthDimension(),
            ]
        else:
            dimensions = [
                BatchDimension(),
                ImageHeightDimension(),
                ImageWidthDimension(),
                ImageChannelDimension(channel_format="bgr"),
            ]
        super().__init__(dimensions)


class BinarySegmentationMasks(TensorSemantics):
    """BinarySegmentationMasks represents the output of a segmentation model with binary masks.

    The expected tensor shape is [B, NUM_CLASSES, H, W].
    For example, a segmentation model with 4 classes would output a tensor of shape [B, 4, H, W],
    where each channel is a binary mask per-pixel.
    """

    def __init__(self):
        dimensions = [
            BatchDimension(),
            SegmentationMaskChannel("bool_masks"),
            ImageHeightDimension(),
            ImageWidthDimension(),
        ]
        super().__init__(dimensions)


class YOLOOutput(TensorSemantics):
    """YOLOOutput represents a detection output from a YOLO model.

    If `has_box_conf=True` then the YOLO model should output a tensor of shape [B, NUM_BOXES, 5 + NUM_CLASSES].
    Otherwise, it should output a tensor of shape [B, NUM_BOXES, 4 + NUM_CLASSES].
    """

    def __init__(self, version="v5"):
        if version != "v5" and version != "v8":
            raise ValueError(
                "gml.tensor.YOLOOutput alias currently only supports YOLO versions v5 and v8"
            )
        dimensions = [
            BatchDimension(),
        ]

        if version == "v5":
            dimensions.extend(
                [
                    DetectionNumCandidatesDimension(is_nms=False),
                    DetectionOutputDimension(
                        coordinates_start_index=0,
                        box_format=BoundingBoxFormat(
                            box_format="cxcywh",
                            is_normalized=False,
                        ),
                        box_confidence_index=4,
                        scores_range=(5, -1),
                    ),
                ]
            )
        elif version == "v8":
            dimensions.extend(
                [
                    DetectionOutputDimension(
                        coordinates_start_index=0,
                        box_format=BoundingBoxFormat(
                            box_format="cxcywh",
                            is_normalized=False,
                        ),
                        scores_range=(4, -1),
                    ),
                    DetectionNumCandidatesDimension(is_nms=False),
                ]
            )

        super().__init__(dimensions)
