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
# Modified   : Wednesday August 20th 2025 12:23:39 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Object Database Module"""
from __future__ import annotations

import logging
import pickle
from typing import Any, Callable, Optional, TypeVar, cast

import redis

from mini_transformer.infra.database.dal import DAL

# ------------------------------------------------------------------------------------------------ #
T = TypeVar("T")  # the dataclass (or any picklable) type
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


class ObjectAccessLayer(DAL):
    """Redis-backed Object Access Layer (OAL).

    This implementation stores whole Python objects under Redis keys derived
    from logical names. It is intended as a simple, opaque object store:
    callers interact via names/IDs; the physical storage and serialization
    are encapsulated here.

    Semantics
    ---------
    - ``create`` is **create-only** (no overwrite). Internally uses ``SET NX``.
    - ``read`` returns the stored object or raises ``FileNotFoundError``.
    - ``delete`` is **idempotent**; returns ``True`` if a key was removed.
    - ``exists`` returns a boolean presence check with no side effects.

    Serialization
    -------------
    - Defaults to ``pickle`` (binary). You can supply custom ``serializer`` /
      ``deserializer`` callables to change the wire format (e.g., JSON).
    - For pickle-backed stores, configure your Redis client/connection pool
      with ``decode_responses=False`` to ensure byte-accurate round trips.

    Expiration (optional)
    ---------------------
    - If ``ttl`` (seconds) is provided, ``create`` sets an expiry (``EX=ttl``).
      Omitted/None means no expiration.

    Constructor
    -----------
    The class docstring intentionally documents constructor behavior per Google
    style. Do not add an ``__init__`` docstring.

    Args:
      pool: A ``redis.ConnectionPool`` used to create the Redis client.
      namespace: Prefix applied to all keys, e.g., ``"oal:"``.
      serializer: Callable converting Python objects to ``bytes``.
      deserializer: Callable converting ``bytes`` back to Python objects.
      ttl: Optional time-to-live in seconds for created keys.

    Raises:
      ValueError: If ``serializer``/``deserializer`` are not provided as a pair.
    """

    def __init__(
        self,
        pool: redis.ConnectionPool,
        *,
        namespace: str = "oal:",
        serializer: Optional[Callable[[Any], bytes]] = None,
        deserializer: Optional[Callable[[bytes], Any]] = None,
        ttl: Optional[int] = None,
    ) -> None:
        if (serializer is None) ^ (deserializer is None):
            raise ValueError(
                "serializer and deserializer must be provided as a pair, or neither."
            )

        self._r = redis.Redis(connection_pool=pool)
        self._ns = namespace
        self._ttl = ttl
        self._dumps: Callable[[Any], bytes] = serializer or pickle.dumps
        self._loads: Callable[[bytes], Any] = deserializer or pickle.loads

    # --- DAL interface -----------------------------------------------------

    def create(self, name: str, data: Any) -> None:
        """Create a new object under ``name`` (create-only; no overwrite).

        Args:
          name: Logical object name or identifier.
          data: The Python object to persist.

        Raises:
          FileExistsError: If an object with ``name`` already exists.
          TypeError: If serialization fails due to unsupported type.
          OSError: For Redis communication errors.
        """
        key = self._key(name)
        try:
            payload = self._dumps(data)
        except Exception as e:  # serializer-specific exception types vary
            raise TypeError(f"Failed to serialize object for name='{name}'.") from e

        try:
            ok = self._r.set(name=key, value=payload, nx=True, ex=self._ttl)
        except redis.RedisError as e:
            raise OSError("Redis error during create().") from e

        # redis-py returns True on success, None/False on NX failure
        if not ok:
            msg = f"Unable to create {name}. It already exists."
            logger.error(msg)
            raise FileExistsError(msg)

    def read(self, name: str) -> Any:
        """Retrieve an object by ``name``.

        Args:
          name: Logical object name or identifier.

        Returns:
          The deserialized Python object stored at ``name``.

        Raises:
          FileNotFoundError: If no object with ``name`` exists.
          ValueError: If deserialization fails (corrupted payload).
          OSError: For Redis communication errors.
        """

        key = self._key(name)
        try:
            raw = cast(
                Optional[bytes], self._r.get(key)
            )  # tell the checker it's bytes|None
        except redis.RedisError as e:
            raise OSError("Redis error during read().") from e

        if raw is None:
            raise FileNotFoundError(f"Unable to read {name}. Object does not exist.")

        # runtime guard (catches accidental decode_responses=True)
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise TypeError(
                "Expected bytes; ensure decode_responses=False for pickle payloads."
            )

        return self._loads(bytes(raw))

    def delete(self, name: str) -> bool:
        """Delete the object stored under ``name`` if present.

        Args:
          name: Logical object name or identifier.

        Returns:
          True if a key was removed; False if no matching key existed.

        Raises:
          OSError: For Redis communication errors.
        """
        key = self._key(name)
        try:
            removed = self._r.delete(key)
        except redis.RedisError as e:
            raise OSError("Redis error during delete().") from e
        return bool(removed)

    def exists(self, name: str) -> bool:
        """Check whether an object exists.

        Args:
          name: Logical object name or identifier.

        Returns:
          True if an object with ``name`` exists; otherwise, False.

        Raises:
          OSError: For Redis communication errors.
        """
        key = self._key(name)
        try:
            return bool(self._r.exists(key))
        except redis.RedisError as e:
            raise OSError("Redis error during exists().") from e

    # --- internal helpers --------------------------------------------------

    def _key(self, name: str) -> str:
        """Compose a namespaced Redis key from a logical ``name``."""
        return f"{self._ns}{name}"
