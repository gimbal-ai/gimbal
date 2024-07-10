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


class Pipeline:
    @abc.abstractmethod
    def to_yaml(self, models: List[str], org_name: str) -> str:
        pass


class SingleModelPipeline(Pipeline):
    def to_yaml(self, models: List[str], org_name: str) -> str:
        if len(models) != 1:
            raise ValueError(
                "{} only supports a single model".format(type(self).__qualname__)
            )
        return self._to_yaml(models[0], org_name)

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
- name: frame_metrics_sink
  kind: FrameMetricsSink
  attributes:
    frame_rate_limit: 30
  inputs:
    frame: .camera_source.frame
  outputs:
  - frame_metrics
- name: detection_metrics_sink
  kind: DetectionsMetricsSink
  attributes:
  inputs:
    detections: .detect.detections
- name: pipeline_latency_metrics_sink
  kind: LatencyMetricsSink
  attributes:
    name: model
  inputs:
    reference: .camera_source.frame
    detections: .detect.detections
    frame_metrics: .frame_metrics_sink.frame_metrics
- name: video_stream_sink
  kind: VideoStreamSink
  attributes:
    frame_rate_limit: 30
  inputs:
    frame: .camera_source.frame
    detections: .detect.detections
    frame_metrics: .frame_metrics_sink.frame_metrics
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
- name: frame_metrics_sink
  kind: FrameMetricsSink
  attributes:
    frame_rate_limit: 30
  inputs:
    frame: .camera_source.frame
  outputs:
  - frame_metrics
- name: pipeline_latency_metrics_sink
  kind: LatencyMetricsSink
  attributes:
    name: model
  inputs:
    reference: .camera_source.frame
    segmentation: .segment.segmentation
    frame_metrics: .frame_metrics_sink.frame_metrics
- name: video_stream_sink
  kind: VideoStreamSink
  attributes:
    frame_rate_limit: 30
  inputs:
    frame: .camera_source.frame
    frame_metrics: .frame_metrics_sink.frame_metrics
    segmentation: .segment.segmentation
"""


# editorconfig-checker-enable
