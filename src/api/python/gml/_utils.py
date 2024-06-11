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
