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


class ImageToFloatTensor(ImagePreprocessingStep):
    def __init__(self, scale: bool = True):
        if not scale:
            raise ValueError(
                "ImageToTensor only currently implemented with scaling enabled."
            )
        self.scale = scale

    def to_proto(self) -> modelexecpb.ImagePreprocessingStep:
        return modelexecpb.ImagePreprocessingStep(
            kind=modelexecpb.ImagePreprocessingStep.IMAGE_PREPROCESSING_KIND_CONVERT_TO_TENSOR,
            conversion_params=modelexecpb.ImagePreprocessingStep.ImageConversionParams(
                scale=self.scale
            ),
        )
