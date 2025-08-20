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
# Modified   : Tuesday August 19th 2025 11:18:49 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Data Access Layer: Defines Interface for  Data Access Objects"""

from abc import ABC, abstractmethod
from typing import Any


# ------------------------------------------------------------------------------------------------ #
class DAL(ABC):
    """Abstract base class for a minimal Data Access Layer (DAL).

    The DAL provides a storage-agnostic interface for working with logical
    object names/IDs rather than physical paths. Implementations encapsulate
    all persistence details (e.g., filesystem, S3, database).

    Contract:
      * `create()` is **create-only** and must not overwrite existing objects.
      * `read()` returns the stored object or raises if it is absent.
      * `delete()` is **idempotent** and returns whether anything was removed.
      * `exists()` reports presence of the object without side effects.

    Notes:
      * Payload type is `Any` to remain backend-agnostic; concrete
        implementations should document the accepted data types (e.g., bytes,
        JSON-serializable dicts, streams).
      * The class docstring documents constructor behavior per Google style;
        do not add an `__init__` docstring.
    """

    @abstractmethod
    def create(self, name: str, data: Any) -> None:
        """Create a new object (create-only; no overwrite).

        Args:
          name: Logical object name or identifier.
          data: The value to persist. Accepted types are implementation-specific.

        Raises:
          FileExistsError: If an object with the given ``name`` already exists.
          OSError: For other I/O or permission-related failures.
        """

    @abstractmethod
    def read(self, name: str) -> Any:
        """Retrieve an object by name.

        Args:
          name: Logical object name or identifier.

        Returns:
          The stored value associated with ``name``. The concrete type depends on
          the implementation (e.g., bytes, dict, list).

        Raises:
          FileNotFoundError: If no object with ``name`` exists.
          OSError: For other I/O or permission-related failures.
        """

    @abstractmethod
    def delete(self, name: str) -> bool:
        """Delete an object by name.

        Args:
          name: Logical object name or identifier.

        Returns:
          True if an object was removed; False if no matching object existed.

        Raises:
          OSError: For storage errors other than a simple not-found condition.
        """

    @abstractmethod
    def exists(self, name: str) -> bool:
        """Check whether an object exists.

        Args:
          name: Logical object name or identifier.

        Returns:
          True if an object with ``name`` exists; otherwise, False.
        """
