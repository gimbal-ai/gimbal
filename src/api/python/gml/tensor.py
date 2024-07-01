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

import abc
from typing import List, Literal, Optional, Tuple

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb


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
        coordinates_start_index: int,
        box_format: BoundingBoxFormat,
        box_confidence_index: int,
        class_index: Optional[int] = None,
        scores_range: Optional[Tuple[int, int]] = None,
    ):
        self.coordinates_range = (coordinates_start_index, 4)
        self.box_format = box_format
        self.box_confidence_index = box_confidence_index
        self.class_index = class_index
        self.scores_range = scores_range

    def to_proto(self) -> modelexecpb.DimensionSemantics:
        scores_range = None
        if self.scores_range is not None:
            scores_range = (
                modelexecpb.DimensionSemantics.DetectionOutputParams.IndexRange(
                    start=self.scores_range[0],
                    size=self.scores_range[1],
                )
            )
        return modelexecpb.DimensionSemantics(
            kind=modelexecpb.DimensionSemantics.DIMENSION_SEMANTICS_KIND_DETECTION_OUTPUT,
            detection_output_params=modelexecpb.DimensionSemantics.DetectionOutputParams(
                box_coordinate_range=modelexecpb.DimensionSemantics.DetectionOutputParams.IndexRange(
                    start=self.coordinates_range[0],
                    size=self.coordinates_range[1],
                ),
                box_format=self.box_format.to_proto(),
                box_confidence_index=self.box_confidence_index,
                class_index=self.class_index,
                scores_range=scores_range,
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
        case "score_masks":
            return (
                modelexecpb.DimensionSemantics.SegmentationMaskParams.SEGMENTATION_MASK_KIND_SCORE
            )
        case _:
            raise ValueError("Invalid segmentation mask kind: {}".format(kind))


class SegmentationMaskChannel(DimensionSemantics):
    def __init__(self, kind: Literal["bool_masks", "int_label_masks", "score_masks"]):
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


class TensorSemantics:
    def __init__(self, dimensions: List[DimensionSemantics]):
        self.dimensions = dimensions

    def to_proto(self) -> modelexecpb.TensorSemantics:
        return modelexecpb.TensorSemantics(
            dimensions=[dim.to_proto() for dim in self.dimensions],
        )
