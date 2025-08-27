#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/infra/dal/oal.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 18th 2025 11:59:17 pm                                                 #
# Modified   : Wednesday August 27th 2025 12:16:11 pm                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Object Database Module"""
from __future__ import annotations

import os
import shelve
from pathlib import Path
from typing import Any, List


# ------------------------------------------------------------------------------------------------ #
class ObjectAccessLayer:
    def __init__(self, location: str) -> None:
        super().__init__()
        self._location = location
        self._path = Path(os.path.join(location, "oal"))
        self._path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def location(self) -> str:
        return self._location

    def create(self, key: str, data: Any) -> None:
        """Create a new entry. Raises if the key already exists."""
        with self._open() as db:
            if key in db:
                raise KeyError(f"Key {key!r} already exists")
            db[key] = data  # closing the shelf flushes

    def read(self, key: str) -> Any:
        """Return the object stored under `key`. Raises KeyError if missing."""
        try:
            with self._open() as db:
                return db[key]
        except KeyError:
            raise KeyError(f"Key {key!r} not found") from None
        except OSError as e:
            raise RuntimeError(f"Failed to open shelf at {self._path}") from e

    def delete(self, key: str) -> bool:
        """Delete the entry; return True if it existed, False otherwise."""
        try:
            with self._open() as db:
                if key in db:
                    del db[key]
                    return True
                return False
        except OSError as e:
            raise RuntimeError(f"Failed to open shelf at {self._path}") from e

    def get_all_names(self) -> List[str]:
        """Return all logical keys stored (dataset_ids)."""
        try:
            with self._open() as db:
                return list(db.keys())
        except OSError:
            return []

    def exists(self, key: str) -> bool:
        """Return True if `key` exists (does not create a new shelf)."""
        try:
            with self._open() as db:
                return key in db
        except OSError:
            return False

    def _open(self, *, flag: str = "c"):
        # writeback=False by default; protocol=None = highest available
        return shelve.open(self._path, flag=flag, protocol=None)
