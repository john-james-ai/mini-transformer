#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /dataset.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 08:17:31 am                                                #
# Modified   : Tuesday August 19th 2025 04:15:03 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #

from collections.abc import Iterator
from typing import Any, Dict, List

from mini_transformer.data.config import DatasetConfig


# ------------------------------------------------------------------------------------------------ #
class Dataset(Iterator):
    """Single-pass in-memory dataset iterator (language-agnostic).

    This iterator yields one record at a time from a pre-materialized list of
    examples using normalized keys ``"src"`` and ``"tgt"`` for source and
    target text, regardless of underlying language codes. It is intentionally
    minimal for toy or small-scale experiments: iteration is single-pass
    (advances an internal index) and the data are held entirely in memory.

    Example:
        >>> cfg = DatasetConfig()
        >>> data = [{"src": "hello", "tgt": "bonjour"},
        ...         {"src": "bye",   "tgt": "au revoir"}]
        >>> ds = Dataset(cfg, data)
        >>> len(ds)
        2
        >>> for row in ds:
        ...     print(row["src"], "->", row["tgt"])
        hello -> bonjour
        bye -> au revoir

    Note:
        The actual source/target language information (e.g., ``en``/``fr``)
        is carried by ``DatasetConfig`` (e.g., ``lang_src``/``lang_tgt``) or
        your run metadata, not per-row fields. Keep rows minimal and uniform.

    Attributes:
        _config (DatasetConfig): The dataset configuration used to materialize
            or describe the records. Stored for provenance; not used to
            transform rows.
        _data (list[dict[str, Any]]): The in-memory sequence of examples,
            each containing at least ``"src"`` and ``"tgt"`` keys.
        _idx (int): Zero-based cursor indicating the next item to yield.
    """

    def __init__(self, config: DatasetConfig, data: List[Dict[str, Any]]) -> None:
        """Initialize the dataset.

        Args:
            config: Configuration describing the dataset sample (e.g., source,
                split, sampling parameters).
            data: Materialized records to iterate over. Each item is typically
                a mapping like ``{"src": "...", "tgt": "...", "split": "train"}``.

        """
        self._config = config
        self._data = data
        self._idx = 0

    def __len__(self) -> int:
        """Return the number of records.

        Returns:
            int: The total number of items available in this dataset.
        """
        return len(self._data)

    def __next__(self) -> Dict[str, Any]:
        """Return the next record in the sequence.

        Returns:
            dict[str, Any]: The next example (must include ``"src"`` and ``"tgt"``).

        Raises:
            StopIteration: If the iterator is exhausted.
        """
        if self._idx < len(self._data):
            row = self._data[self._idx]
            self._idx += 1
            return row
        else:
            raise StopIteration

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Retrieve a record by absolute index.

        Args:
            idx (int): Zero-based position of the item to access. Negative indices
                are supported, consistent with standard Python sequence semantics
                (e.g., ``-1`` refers to the last item).

        Returns:
            Optional[dict[str, Any]]: The record at ``idx`` containing at least
            ``"src"`` and ``"tgt"`` keys.

        Raises:
            IndexError: If ``idx`` is outside the valid range
                ``[-len(self), len(self) - 1]``.

        Notes:
            This method delegates to the underlying list, so it raises
            ``IndexError`` on out-of-range indices rather than returning ``None``.
            If you prefer a ``None`` return instead, catch ``IndexError`` inside
            this method and return ``None``.
        """
        return self._data[idx]
