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
import tempfile
from pathlib import Path
from typing import Dict


class AssetManager:
    @abc.abstractmethod
    def add_asset(self, name: str) -> Path:
        pass

    @abc.abstractmethod
    def assets(self) -> Dict[str, Path]:
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc, value, tb):
        pass


class DirectoryAssetManager(AssetManager):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._asset_paths: Dict[str, Path] = dict()

    def add_asset(self, name: str) -> Path:
        path = self.path / name
        self._asset_paths[name] = path
        return path

    def assets(self) -> Dict[str, Path]:
        return self._asset_paths


class TempFileAssetManager(AssetManager):
    def __init__(self):
        self._assets = dict()
        self._asset_paths = dict()

    def add_asset(self, name: str) -> Path:
        tmp = tempfile.NamedTemporaryFile(mode="w")
        self._assets[name] = tmp
        file = tmp.__enter__()
        self._asset_paths[name] = Path(file.name)
        return self._asset_paths[name]

    def assets(self) -> Dict[str, Path]:
        return self._asset_paths

    def __enter__(self):
        return self

    def __exit__(self, exc, value, tb):
        for tmp in self._assets.values():
            tmp.__exit__(exc, value, tb)
        self._assets.clear()
        self._asset_paths.clear()
