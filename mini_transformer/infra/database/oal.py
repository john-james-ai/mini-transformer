#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /oal.py                                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 18th 2025 11:59:17 pm                                                 #
# Modified   : Friday August 22nd 2025 06:20:59 am                                                 #
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
        self._path = Path(os.path.join(location, "oal"))
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def _open(self, *, flag: str = "c"):
        # writeback=False by default; protocol=None = highest available
        return shelve.open(self._path, flag=flag, protocol=None)

    def create(self, name: str, data: Any) -> None:
        """Create a new entry. Raises if the name already exists."""
        with self._open(flag="c") as db:
            if name in db:
                raise KeyError(f"Key {name!r} already exists")
            db[name] = data  # closing the shelf flushes

    def read(self, name: str) -> Any:
        """Return the object stored under `name`. Raises KeyError if missing."""
        try:
            with self._open(flag="r") as db:
                return db[name]
        except KeyError:
            raise KeyError(f"Key {name!r} not found") from None
        except OSError as e:
            raise RuntimeError(f"Failed to open shelf at {self._path}") from e

    def delete(self, name: str) -> bool:
        """Delete the entry; return True if it existed, False otherwise."""
        try:
            with self._open(flag="c") as db:
                if name in db:
                    del db[name]
                    return True
                return False
        except OSError as e:
            raise RuntimeError(f"Failed to open shelf at {self._path}") from e

    def get_all_names(self) -> List[str]:
        """Return all logical keys stored (dataset_ids)."""
        try:
            with self._open(flag="r") as db:
                return list(db.keys())
        except OSError:
            return []

    def exists(self, name: str) -> bool:
        """Return True if `name` exists (does not create a new shelf)."""
        try:
            with self._open(flag="r") as db:
                return name in db
        except OSError:
            return False
