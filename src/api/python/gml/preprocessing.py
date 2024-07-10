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
