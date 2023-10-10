#!/bin/bash -ex

# Copyright 2018- The Pixie Authors.
# Modifications Copyright 2023- Gimlet Labs, Inc.
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

tot=$(git rev-parse --show-toplevel)

# TODO(james/philkuz): replace this script with a bazel version when the node rules are ready.
pushd "${tot}/src/ui" >/dev/null
pnpm install
# Note: this _should_ be schema.d.ts, not schema.ts. However, since we build the UI in isolatedModules mode, we must
# also use normal enums instead of const enums (see https://www.typescriptlang.org/tsconfig#isolatedModules).
# A strict declarations file (.d.ts) can't export values (like normal enums), so we must rename the file as well.
pnpm exec graphql-schema-typescript generate-ts "${tot}/src/controlplane/api/gql/schema" --output app/graphql-schema.ts
sed -i 's/export const enum/export enum/g' app/graphql-schema.ts
popd >/dev/null
