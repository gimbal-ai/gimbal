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


def prepare_ultralytics_yolo(model):
    """Prepares an ultralytics YOLO model for export.

    Ultralytics YOLO models requires setting `export=True` on some of the torch modules for exporting to work properly.
    This function handles setting that value on the necessary modules.
    """
    if not hasattr(model, "model"):
        raise ValueError(
            "input to `prepare_ultralytics_yolo` is not a supported ultralytics yolo model"
        )
    if hasattr(model.model, "fuse") and callable(model.model.fuse):
        model.model.fuse()

    for _, m in model.named_modules():
        if hasattr(m, "export"):
            m.export = True
