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

load("@bazel_gazelle//:def.bzl", "gazelle")

_THIRD_PARTY_IMPORTS = [
    "//third_party/github.com/open-telemetry/opentelemetry-proto:imports.csv",
    "//third_party/github.com/gogo/protobuf:imports.csv",
    "//third_party/github.com/google/mediapipe:imports.csv",
    "//third_party/github.com/qdrant/go-client:imports.csv",
]

_GAZELLE_COMMON_ATTRS = dict(
    args = [
        "-proto_configs=$(location //bazel:rules_proto_config.yaml)",
        "-proto_imports_in={}".format(
            ",".join([
                "$(location {})".format(label)
                for label in _THIRD_PARTY_IMPORTS
            ]),
        ),
    ],
    data = [
        "//bazel:rules_proto_config.yaml",
    ] + _THIRD_PARTY_IMPORTS,
    prefix = "gimletlabs.ai/gimlet",
)

def _gazelle_runners(**kwargs):
    attrs = dict(_GAZELLE_COMMON_ATTRS, **kwargs)
    gazelle(
        name = "gazelle",
        command = "fix",
        mode = "fix",
        **attrs
    )

    gazelle(
        name = "gazelle_diff",
        command = "fix",
        mode = "diff",
        **attrs
    )

gazelle_runners = _gazelle_runners
