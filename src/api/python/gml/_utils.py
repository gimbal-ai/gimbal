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

import hashlib
import os
from typing import BinaryIO, TextIO


def chunk_file(f: TextIO | BinaryIO, chunk_size=64 * 1024):
    while True:
        chunk = f.read(chunk_size)
        if not chunk:
            break
        yield chunk


def sha256sum(f: BinaryIO, buffer_size=64 * 1024):
    sha256sum = hashlib.sha256()
    for chunk in chunk_file(f, buffer_size):
        sha256sum.update(chunk)

    return sha256sum.hexdigest()


def get_file_size(f: TextIO | BinaryIO):
    f.seek(0, os.SEEK_END)
    file_size = f.tell()
    f.seek(0)
    return file_size
