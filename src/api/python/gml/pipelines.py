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
from gml.model import Model


class Pipeline:
    @abc.abstractmethod
    def to_yaml(self, models: List[Model], org_name: str) -> str:
        pass


class SingleModelPipeline(Pipeline):
    def to_yaml(self, models: List[Model], org_name: str) -> str:
        if len(models) != 1:
            raise ValueError(
                "{} only supports a single model".format(type(self).__qualname__)
            )
        return self._to_yaml(models[0].name, org_name)

    @abc.abstractmethod
    def _to_yaml(self, model_name: str, org_name: str) -> str:
        pass


class SimpleDetectionPipeline(SingleModelPipeline):
    def __init__(self, add_tracking_id: bool = False):
        self.add_tracking_id = add_tracking_id

    def _to_yaml(self, model_name: str, org_name: str):
        add_tracking_id = "true" if self.add_tracking_id else "false"
        # editorconfig-checker-disable
        return f"""---
nodes:
- name: camera_source
  kind: CameraSource
  outputs:
  - frame
- name: detect
  kind: Detect
  attributes:
    add_tracking_id: {add_tracking_id}
    model:
      model:
        name: {model_name}
        org: {org_name}
    frame_rate_limit: 30
  inputs:
    frame: .camera_source.frame
  outputs:
  - detections
- name: video_stream_sink
  kind: VideoStreamSink
  attributes:
    frame_rate_limit: 30
  inputs:
    frame: .camera_source.frame
    detections: .detect.detections
"""


# editorconfig-checker-enable


class SimpleSegmentationPipeline(SingleModelPipeline):
    def _to_yaml(self, model_name: str, org_name: str):
        # editorconfig-checker-disable
        return f"""---
nodes:
- name: camera_source
  kind: CameraSource
  outputs:
  - frame
- name: segment
  kind: Segment
  attributes:
    model:
      model:
        name: {model_name}
        org: {org_name}
    frame_rate_limit: 30
  inputs:
    frame: .camera_source.frame
  outputs:
  - segmentation
- name: video_stream_sink
  kind: VideoStreamSink
  attributes:
    frame_rate_limit: 30
  inputs:
    frame: .camera_source.frame
    segmentation: .segment.segmentation
"""


class LiveChatPipeline(Pipeline):
    def to_yaml(self, models: List[Model], org_name: str) -> str:
        if len(models) != 2:
            raise ValueError(
                "LiveChatPipeline expects two models (a tokenizer and a language model)"
            )
        tokenizer = None
        lm = None
        for m in models:
            if m.storage_format == modelexecpb.ModelInfo.MODEL_STORAGE_FORMAT_OPAQUE:
                tokenizer = m
            if m.generation_config is not None:
                lm = m
        if tokenizer is None or lm is None:
            raise ValueError(
                "LiveChatPipeline expects both a tokenizer model and a language model)"
            )
        return f"""---
nodes:
- name: text_source
  kind: TextStreamSource
  outputs:
  - prompt
- name: tokenize
  kind: Tokenize
  attributes:
    tokenizer:
      model:
        name: {tokenizer.name}
        org: {org_name}
  inputs:
    text: .text_source.prompt
  outputs:
  - tokens
- name: generate
  kind: GenerateTokens
  attributes:
    model:
      model:
        name: {lm.name}
        org: {org_name}
  inputs:
    prompt: .tokenize.tokens
  outputs:
  - generated_tokens
- name: detokenize
  kind: Detokenize
  attributes:
    tokenizer:
      model:
        name: {tokenizer.name}
        org: {org_name}
  inputs:
    tokens: .generate.generated_tokens
  outputs:
  - text
- name: text_sink
  kind: TextStreamSink
  inputs:
    text_batch: .detokenize.text
"""


# editorconfig-checker-enable
