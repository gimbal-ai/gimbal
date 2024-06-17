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
from typing import List

import gml.proto.src.api.corepb.v1.model_exec_pb2 as modelexecpb


class ImagePreprocessingStep(abc.ABC):
    @abc.abstractmethod
    def to_proto(self) -> modelexecpb.ImagePreprocessingStep:
        pass


class LetterboxImage(ImagePreprocessingStep):
    def to_proto(self) -> modelexecpb.ImagePreprocessingStep:
        return modelexecpb.ImagePreprocessingStep(
            kind=modelexecpb.ImagePreprocessingStep.IMAGE_PREPROCESSING_KIND_RESIZE,
            resize_params=modelexecpb.ImagePreprocessingStep.ImageResizeParams(
                kind=modelexecpb.ImagePreprocessingStep.ImageResizeParams.IMAGE_RESIZE_KIND_LETTERBOX,
            ),
        )


class ResizeImage(ImagePreprocessingStep):
    """ResizeImage resizes the image to the target size without preserving aspect ratio."""

    def to_proto(self) -> modelexecpb.ImagePreprocessingStep:
        return modelexecpb.ImagePreprocessingStep(
            kind=modelexecpb.ImagePreprocessingStep.IMAGE_PREPROCESSING_KIND_RESIZE,
            resize_params=modelexecpb.ImagePreprocessingStep.ImageResizeParams(
                kind=modelexecpb.ImagePreprocessingStep.ImageResizeParams.IMAGE_RESIZE_KIND_STRETCH,
            ),
        )


class StandardizeTensor(ImagePreprocessingStep):
    """StandardizeTensor standardizes the tensor with the given means and standard deviations."""

    def __init__(self, means: List[float], stddevs: List[float]):
        self.means = means
        self.stddevs = stddevs

    def to_proto(self) -> modelexecpb.ImagePreprocessingStep:
        return modelexecpb.ImagePreprocessingStep(
            kind=modelexecpb.ImagePreprocessingStep.IMAGE_PREPROCESSING_KIND_STANDARDIZE,
            standardize_params=modelexecpb.ImagePreprocessingStep.ImageStandardizeParams(
                means=self.means,
                stddevs=self.stddevs,
            ),
        )


class ImageToFloatTensor(ImagePreprocessingStep):
    def __init__(self, scale: bool = True):
        self.scale = scale

    def to_proto(self) -> modelexecpb.ImagePreprocessingStep:
        return modelexecpb.ImagePreprocessingStep(
            kind=modelexecpb.ImagePreprocessingStep.IMAGE_PREPROCESSING_KIND_CONVERT_TO_TENSOR,
            conversion_params=modelexecpb.ImagePreprocessingStep.ImageConversionParams(
                scale=self.scale
            ),
        )
