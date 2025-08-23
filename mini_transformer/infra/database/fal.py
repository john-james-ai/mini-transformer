#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /fal.py                                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 18th 2025 11:59:17 pm                                                 #
# Modified   : Friday August 22nd 2025 08:14:54 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""File Access Layer"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List

from mini_transformer.infra.database.dal import DAL

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
class FileAccessLayer(DAL):
    """File-backed data access layer that reads/writes JSON Lines (JSONL).

    Each stored object is represented as a list of dictionaries and is persisted
    to a single file where each line is one JSON object. Paths are provided via
    the `key` parameter and are interpreted as filesystem paths.

    Notes:
        * Files are written with UTF-8 encoding and Unix newlines (`\\n`).
        * Creation uses exclusive mode (`"x"`) to avoid overwriting existing files.
        * JSON is written with `ensure_ascii=False` to preserve non-ASCII characters.
    """

    def __init__(self, location: str) -> None:
        super().__init__()
        self._location = location

    def create(self, key: str, data: List[Dict[str, str]]) -> None:
        """Create a new JSONL file at `key` from a list of row dicts.

        Writes one JSON object per line. Parent directories are created if needed.
        Creation is exclusive; if the file already exists, a `FileExistsError` is raised.

        Args:
            key: Filesystem path where the JSONL file will be created.
            data: List of row dictionaries to serialize (one per line).

        Raises:
            FileExistsError: If a file already exists at `key`.
            OSError: For other I/O errors encountered during write.
        """
        filepath = self._get_filepath(key=key)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(filepath, "x", encoding="utf-8", newline="\n") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except FileExistsError:
            msg = f"Unable to create {key}. It already exists."
            logger.error(msg)
            raise FileExistsError(msg)

    def read(self, key: str) -> List[Dict[str, str]]:
        """Read a JSONL file from `key` into a list of row dicts.

        Args:
            key: Filesystem path of the JSONL file to read.

        Returns:
            A list of dictionaries, one per JSON line.

        Raises:
            FileNotFoundError: If no file exists at `key`.
            ValueError: If a line cannot be parsed as valid JSON.
            OSError: For other I/O errors encountered during read.
        """
        filepath = self._get_filepath(key=key)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            msg = f"Unable to read {key}. File does not exist."
            logger.error(msg)
            raise FileNotFoundError(msg)

    def delete(self, key: str) -> bool:
        """Delete the file at `key`.

        This operation is idempotent.

        Args:
            key: Filesystem path of the JSONL file to delete.

        Returns:
            True if the file existed and was deleted, False if it did not exist.

        Raises:
            OSError: For other I/O errors encountered during deletion.
        """
        filepath = self._get_filepath(key=key)
        try:
            os.remove(filepath)
            return True
        except FileNotFoundError as e:
            logger.debug(e)
            return False

    def exists(self, key: str) -> bool:
        """Return whether a file exists at `key`.

        Args:
            key: Filesystem path to check.

        Returns:
            True if a file exists at `key`; False otherwise.
        """
        filepath = self._get_filepath(key=key)
        return os.path.exists(filepath)

    def _get_filepath(self, key: str) -> Path:
        filename = f"{key}.jsonl"
        return Path(os.path.join(self._location, filename))
