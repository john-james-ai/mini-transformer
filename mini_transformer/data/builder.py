#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /builder.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Wednesday August 20th 2025 02:14:01 am                                              #
# Modified   : Wednesday August 20th 2025 06:34:10 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import random
from abc import ABC, abstractmethod
from hashlib import sha1
from typing import Any, Dict, List

from datasets import IterableDataset, load_dataset

from mini_transformer.data.config import DatasetConfig
from mini_transformer.data.dataset import Dataset, TranslationDataset


# ------------------------------------------------------------------------------------------------ #
class DatasetBuilder(ABC):
    """Abstract base for dataset builders.

    Subclasses implement concrete extraction and construction logic for specific
    dataset types (e.g., translation pairs, classification records).

    Methods:
        build: Construct and return a concrete `Dataset` instance.
    """

    @abstractmethod
    def build(self) -> Dataset:
        """Build the dataset.

        Returns:
            Dataset: A concrete dataset instance ready for downstream use
            (e.g., training, evaluation).

        Raises:
            ValueError: If the builder cannot construct a dataset due to
                insufficient valid examples or configuration issues.
        """


# ------------------------------------------------------------------------------------------------ #
class TranslationDatasetBuilder(DatasetBuilder):
    """Builder for small, filtered translation datasets.

    This class streams a translation dataset from Hugging Face Datasets,
    applies lightweight quality/length filters, oversamples to build a candidate
    pool, and then deterministically subsamples to a fixed size using a seeded RNG.

    Attributes:
        _config (DatasetConfig): Dataset and filtering configuration.
        _dataset: Reserved for future use; not currently populated.
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the builder.

        Args:
            config (DatasetConfig): Configuration that specifies dataset identity,
                language keys, split, size target, oversample factor, min/max token
                thresholds, ratio constraints, and RNG seed.
        """
        self._config = config
        self._dataset = None

    def build(self) -> Dataset:
        """Construct a `TranslationDataset`.

        The pipeline is:
        1) Stream the source split.
        2) Extract and filter candidate pairs until the oversampled target is reached.
        3) Deterministically sample `dataset_size` examples using `seed`.

        Returns:
            Dataset: A `TranslationDataset` containing exactly
            `config.dataset_size` (src, tgt) pairs.

        Raises:
            ValueError: If there are not enough valid rows to satisfy
                `config.dataset_size` after filtering.
        """
        stream = self._load_dataset()
        data = self._extract(data=stream)
        return TranslationDataset(config=self._config, data=data)

    def _load_dataset(self) -> IterableDataset:
        """Load the dataset stream from Hugging Face Datasets.

        Uses `datasets.load_dataset(..., streaming=True)`.

        Returns:
            IterableDataset: A streaming iterable over the configured split.
        """
        return load_dataset(
            self._config.dataset,
            self._config.lang,
            split=self._config.split,
            streaming=True,
            # type: ignore[reportArgumentType]
        )

    def _extract(self, data: IterableDataset) -> List[Dict[str, Any]]:
        """Stream, filter, and deterministically select translation pairs.

        This method:
        - Streams rows and normalizes each to a flat `{"src": ..., "tgt": ...}` view.
        - Applies `_is_valid_row` to enforce token-length and length-ratio constraints.
        - Collects up to `dataset_size * oversample` candidates.
        - Returns a deterministic sample of size `dataset_size` using `seed`.

        Args:
            data (IterableDataset): Streaming dataset iterator.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries with keys
            `{"idx", "src", "tgt"}` of length `config.dataset_size`.

        Raises:
            ValueError: If the number of valid candidates is smaller than
                `config.dataset_size`.
        """
        candidates: List[Dict[str, Any]] = []
        target = self._config.dataset_size * self._config.oversample

        for row in data:
            candidate = row.get("translation", row)
            if self._is_valid_row(candidate):
                candidates.append(self._parse_row(candidate))
                if len(candidates) >= target:
                    break

        if len(candidates) < self._config.dataset_size:
            raise ValueError(
                f"Not enough valid rows: {len(candidates)} < {self._config.dataset_size}. "
                f"Increase oversample or relax filters."
            )

        # Deterministic selection
        rng = random.Random(self._config.seed)
        return rng.sample(candidates, self._config.dataset_size)

    def _is_valid_row(self, row: Dict[str, Any]) -> bool:
        """Validate a (src, tgt) pair against length and ratio constraints.

        The check enforces:
        - Both sides are non-empty after splitting on whitespace.
        - Token counts satisfy `tokens_min` and per-side max thresholds
          (`src_max_words`, `tgt_max_words`).
        - The length ratio passes `_is_valid_ratio`.

        Args:
            row (Dict[str, Any]): A record containing source and target strings,
                typically under language keys like `"en"` and `"fr"`.

        Returns:
            bool: True if the row passes all filters; False otherwise.
        """
        src_len = len(row[self._config.lang_src].split())
        tgt_len = len(row[self._config.lang_tgt].split())

        if src_len == 0 or tgt_len == 0:
            return False

        return (
            self._is_valid_ratio(src_len=src_len, tgt_len=tgt_len)
            and self._config.tokens_min <= src_len <= self._config.src_max_words
            and self._config.tokens_min <= tgt_len <= self._config.tgt_max_words
        )

    def _is_valid_ratio(self, src_len: int, tgt_len: int) -> bool:
        """Check symmetric length ratio between source and target.

        Uses a symmetric ratio defined as `max(src_len, tgt_len) / min(src_len, tgt_len)`
        and compares against `config.ratio_max`.

        Args:
            src_len (int): Token count of the source text.
            tgt_len (int): Token count of the target text.

        Returns:
            bool: True if the symmetric ratio is within the allowed maximum; otherwise False.

        Notes:
            If you prefer asymmetric bounds, switch to
            `ratio_min <= src_len / tgt_len <= ratio_max`.
        """
        r = max(src_len, tgt_len) / min(src_len, tgt_len)
        return r <= self._config.ratio_max

    def _parse_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw row into a stable `(idx, src, tgt)` mapping.

        The `idx` is computed as the first 12 hex chars of SHA1 over the pair
        `{"src": src, "tgt": tgt}` to provide a compact, deterministic identifier.

        Args:
            row (Dict[str, Any]): A mapping with language keys containing the raw
                source and target strings.

        Returns:
            Dict[str, Any]: A dictionary with keys:
                - `idx` (str): Stable 12-hex identifier.
                - `src` (str): Source text.
                - `tgt` (str): Target text.
        """
        src = row[self._config.lang_src].strip()
        tgt = row[self._config.lang_tgt].strip()
        idx = sha1(repr({"src": src, "tgt": tgt}).encode("utf-8")).hexdigest()[:12]
        return {"idx": idx, "src": src, "tgt": tgt}
