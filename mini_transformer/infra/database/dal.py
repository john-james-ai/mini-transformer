#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /dal.py                                                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 09:53:28 pm                                                #
# Modified   : Friday August 22nd 2025 02:15:31 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Data Access Layer: Defines Interface for  Data Access Objects"""

from abc import ABC, abstractmethod
from typing import Any


# ------------------------------------------------------------------------------------------------ #
class DAL(ABC):
    """Abstract data access layer (DAL) interface.

    This interface defines minimal CRUD-style operations for a storage backend.
    Each method operates on a backend-specific **locator** (`loc`), which must
    uniquely identify a stored object for that backend (e.g., a Redis key or a
    filesystem path). The concrete implementation decides how `data` is
    serialized/persisted.

    Conventions:
      * `create()` should fail if an object already exists at `loc`.
      * `read()` should raise `KeyError` if no object exists at `loc`.
      * `delete()` should be idempotent and return whether a deletion occurred.
      * `exists()` should return whether an object is present at `loc`.
    """

    @abstractmethod
    def create(self, loc: str, data: Any) -> None:
        """Create a new object at `loc`.

        Implementations should persist `data` under the given locator. If an
        object already exists at `loc`, they should raise `KeyError`.

        Args:
            loc: Backend-specific locator (e.g., key or file path).
            data: Object to persist (type is backend-specific but must be
                supported by the implementation’s serialization).

        Raises:
            KeyError: If an object already exists at `loc`.
            Exception: Implementations may raise backend-specific errors
                (e.g., I/O errors) on failure to persist.
        """
        ...

    @abstractmethod
    def read(self, loc: str) -> Any:
        """Read and return the object stored at `loc`.

        Args:
            loc: Backend-specific locator (e.g., key or file path).

        Returns:
            The deserialized object stored at `loc`.

        Raises:
            KeyError: If no object exists at `loc`.
            Exception: Implementations may raise backend-specific errors
                (e.g., I/O errors) on failure to read/deserialize.
        """
        ...

    @abstractmethod
    def delete(self, loc: str) -> bool:
        """Delete the object at `loc`.

        Implementations should be idempotent.

        Args:
            loc: Backend-specific locator (e.g., key or file path).

        Returns:
            True if an object existed and was deleted; False if no object
            existed at `loc`.
        """
        ...

    @abstractmethod
    def exists(self, loc: str) -> bool:
        """Return whether an object exists at `loc`.

        Args:
            loc: Backend-specific locator (e.g., key or file path).

        Returns:
            True if an object exists at `loc`; False otherwise.
        """
        ...
