#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/utils/mixins.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 02:45:28 pm                                                #
# Modified   : Monday August 25th 2025 01:22:02 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""A collection of mixins used in this package."""
from __future__ import annotations

import json
import logging
from dataclasses import fields, is_dataclass
from datetime import datetime
from hashlib import sha1
from typing import Any, Dict, Union

import pandas as pd

from mini_transformer.utils.dtypes import IMMUTABLE_TYPES, SEQUENCE_TYPES

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #


class FingerprintMixin:
    _fp_version = 1

    @property
    def fingerprint(self) -> str:
        if not is_dataclass(self):
            msg = "FingerprintMixin requires a dataclass subclass."
            logger.error(msg)
            raise TypeError(msg)
        payload = {
            f.name: getattr(self, f.name)
            for f in fields(self)  # type: ignore[reportArgumentType]
            if f.metadata.get("stable")
        }
        payload["_v"] = self._fp_version
        s = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return sha1(s.encode("utf-8")).hexdigest()[:8]


# ------------------------------------------------------------------------------------------------ #
class FreezableMixin:
    """Instance-level freezing mixin.

    This mixin lets you create objects that start mutable and can be
    *frozen on demand* via :meth:`freeze`. Once frozen, attempts to set
    or delete attributes raise :class:`AttributeError`. This is useful
    when you want dataclass-like ergonomics without committing to
    ``frozen=True`` at class-definition time.

    Notes:
        - Only attribute assignment/deletion is blocked. Contained
          mutable values (e.g., lists, dicts) remain mutable unless
          you explicitly deep-freeze them before/while calling
          :meth:`freeze`.
        - Works with regular classes and dataclasses. When used with a
          dataclass, prefer calling :meth:`freeze` after construction.
        - The internal flag is stored under ``_is_frozen`` to minimize
          collisions; you can change the constant if needed.

    Attributes:
        _FREEZE_FLAG (str): Name of the internal attribute that stores
            the frozen state as a boolean.
    """

    _FREEZE_FLAG = "_is_frozen"

    def freeze(self) -> "FreezableMixin":
        """Freeze the instance, preventing further attribute mutation.

        Sets an internal boolean flag that causes :meth:`__setattr__`
        and :meth:`__delattr__` to raise :class:`AttributeError` for any
        attribute other than the flag itself.

        Returns:
            FreezableMixin: ``self`` (to allow chaining).

        Examples:
            >>> @dataclass
            ... class C(FreezableMixin):
            ...     x: int
            ...
            >>> c = C(1).freeze()
            >>> c.x
            1
            >>> c.x = 2
            Traceback (most recent call last):
                ...
            AttributeError: C is frozen; cannot set 'x'
        """
        object.__setattr__(self, self._FREEZE_FLAG, True)
        return self

    @property
    def frozen(self) -> bool:
        """Whether the instance is currently frozen.

        Returns:
            bool: ``True`` if the instance is frozen, ``False`` otherwise.
        """
        return getattr(self, self._FREEZE_FLAG, False)

    def __setattr__(self, name: str, value: Any) -> None:
        """Block attribute assignment when the instance is frozen.

        Args:
            name: Attribute name.
            value: New value.

        Raises:
            AttributeError: If attempting to set any attribute while frozen.
        """
        if name != self._FREEZE_FLAG and getattr(self, self._FREEZE_FLAG, False):
            raise AttributeError(
                f"{type(self).__name__} is frozen; cannot set {name!r}"
            )
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        """Block attribute deletion when the instance is frozen.

        Args:
            name: Attribute name to delete.

        Raises:
            AttributeError: If attempting to delete any attribute while frozen.
        """
        if name != self._FREEZE_FLAG and getattr(self, self._FREEZE_FLAG, False):
            raise AttributeError(
                f"{type(self).__name__} is frozen; cannot delete {name!r}"
            )
        object.__delattr__(self, name)


# ------------------------------------------------------------------------------------------------ #
class ObjectRepresentationMixin:
    """Base Class for Data Transfer Objects"""

    def __repr__(self) -> str:
        return "{}({})".format(
            self.__class__.__name__,
            ", ".join(
                "{}={!r}".format(k, v)
                for k, v in self.__dict__.items()
                if type(v) in IMMUTABLE_TYPES
            ),
        )

    def __str__(self) -> str:
        width = 32
        breadth = width * 2
        s = f"\n\n{self.__class__.__name__.center(breadth, ' ')}"
        d = self.as_dict()
        for k, v in d.items():
            if type(v) in IMMUTABLE_TYPES:
                s += f"\n{k.rjust(width,' ')} | {v}"
        s += "\n\n"
        return s

    def as_dict(self) -> Dict[str, Union[str, int, float, datetime, None]]:
        """Returns a dictionary representation of the object."""
        return {
            k: self._export(v)
            for k, v in self.__dict__.items()
            if not k.startswith("_")
        }

    @classmethod
    def _export(
        cls,
        v: Any,
    ) -> Any:  # pragma: no cover
        """Returns v converted to dicts, recursively."""
        if isinstance(v, IMMUTABLE_TYPES):
            return v
        elif isinstance(v, SEQUENCE_TYPES):
            return type(v)(map(cls._export, v))
        elif isinstance(v, dict):
            return v
        elif hasattr(v, "as_dict"):
            return v.as_dict()
        elif isinstance(v, datetime):
            return v.isoformat()
        else:
            return dict()

    def as_df(self) -> Any:
        """Returns the object in DataFrame format"""
        d = self.as_dict()
        return pd.DataFrame(data=d, index=[0])
