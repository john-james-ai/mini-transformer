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
# Modified   : Tuesday August 19th 2025 11:06:51 pm                                                #
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
    """File-backed Data Access Layer using JSON Lines (*.jsonl*) files.

    This implementation treats each logical object name as a single ``.jsonl`` file
    stored under ``data_root``. Writes are **create-only** (exclusive open) to
    preserve immutability; reads are strict (no preprocessing); deletes are
    idempotent (returning ``True`` if removed, ``False`` if the file did not exist).

    Paths:
      - The resolved path is ``<data_root>/<name>.jsonl``.
      - Parent directories are created as needed on ``create()``.

    Error semantics:
      - ``create()`` raises ``FileExistsError`` if the target file already exists.
      - ``read()`` raises ``FileNotFoundError`` if the target file does not exist.
      - ``delete()`` returns ``False`` if the file is missing (and logs the miss).
      - Other I/O errors (permissions, disk issues) propagate.

    Note:
      Per Google style, the class docstring documents constructor behavior; a
      separate ``__init__`` docstring is intentionally omitted.
    """

    def __init__(self, data_root: str) -> None:
        self._data_root = data_root

    def create(self, name: str, data: List[Dict[str, str]]) -> None:
        """Create a new JSONL object (create-only; no overwrite).

        Each item in ``data`` is serialized with ``json.dumps`` and written as a
        single line to ``<data_root>/<name>.jsonl``. Parent directories are created
        as needed. If the file already exists, this method raises.

        Args:
          name: Logical object name (filename stem without extension). May include
            subdirectories (e.g., ``"datasets/en_fr/tokens_v1"``).
          data: A list of row dictionaries to write, one JSON object per line.

        Raises:
          FileExistsError: If ``<name>.jsonl`` already exists.
          OSError: For other filesystem errors (e.g., permission denied).
        """
        filepath = self._get_filepath(name=name)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(filepath, "x", encoding="utf-8", newline="\n") as f:
                for row in data:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        except FileExistsError:
            msg = f"Unable to create {name}. It already exists."
            logger.error(msg)
            raise FileExistsError(msg)

    def read(self, name: str) -> List[Dict[str, str]]:
        """Read a JSONL object into memory.

        The file is read line-by-line and each line is parsed with ``json.loads``.
        No trimming or preprocessing is performed.

        Args:
          name: Logical object name (filename stem without extension).

        Returns:
          The list of row dictionaries parsed from the JSONL file.

        Raises:
          FileNotFoundError: If ``<name>.jsonl`` does not exist.
          json.JSONDecodeError: If a line is not valid JSON.
          OSError: For other filesystem errors.
        """
        filepath = self._get_filepath(name=name)
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                return [json.loads(line) for line in f]
        except FileNotFoundError:
            msg = f"Unable to read {name}. File does not exist."
            logger.error(msg)
            raise FileNotFoundError(msg)

    def delete(self, name: str) -> bool:
        """Delete the JSONL file for ``name`` if it exists.

        Args:
          name: Logical object name (filename stem without extension).

        Returns:
          True if the file was removed; False if the file did not exist.

        Raises:
          OSError: For filesystem errors other than file-not-found.
        """
        filepath = self._get_filepath(name=name)
        try:
            os.remove(filepath)
            return True
        except FileNotFoundError as e:
            logger.debug(e)
            return False

    def exists(self, name: str) -> bool:
        """Check whether ``<name>.jsonl`` exists.

        Args:
          name: Logical object name (filename stem without extension).

        Returns:
          True if the file exists; otherwise False.
        """
        filepath = self._get_filepath(name=name)
        return os.path.exists(filepath)

    def _get_filepath(self, name: str) -> Path:
        """Resolve the filesystem path for ``name``.

        Appends the ``.jsonl`` extension and joins it with ``data_root``.

        Args:
          name: Logical object name (filename stem without extension).

        Returns:
          A ``pathlib.Path`` to ``<data_root>/<name>.jsonl``.
        """
        filename = f"{name}.jsonl"
        return Path(os.path.join(self._data_root, filename))
