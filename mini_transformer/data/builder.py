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
# Modified   : Friday August 22nd 2025 03:23:54 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, Dict, List

from datasets import IterableDataset, load_dataset

from mini_transformer.data.config import DatasetConfig
from mini_transformer.data.dataset import Dataset, TranslationDataset
from mini_transformer.infra.observer import Observer

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TranslationDatasetBuilderObserver(Observer):
    dataset: str = None
    split: str = None
    lang: str = None
    seed: int = None
    dataset_size: int = None
    oversample: int = None

    seen: int = 0
    filtered: int = 0
    candidates: int = 0
    selected: int = 0

    filtered_empty: int = 0
    filtered_ratio: int = 0
    filtered_src_short: int = 0
    filtered_src_long: int = 0
    filtered_tgt_short: int = 0
    filtered_tgt_long: int = 0

    throughput: float = 0.0

    def __str__(self) -> str:
        return self.as_string()

    def end(self) -> None:
        super().end()
        d = self.duration
        self.throughput = round(self.seen / d, 3) if d > 0 else 0.0
        self.freeze()

    def log_summary(self) -> None:
        output = str(self)
        logger.info(output)


# ------------------------------------------------------------------------------------------------ #
class DatasetBuilder(ABC):
    """Abstract base for dataset builders.

    Subclasses implement concrete extraction and construction logic for specific
    dataset types (e.g., translation pairs, classification records).

    Methods:
        build: Construct and return a concrete `Dataset` instance.
    """

    @property
    @abstractmethod
    def observer(self) -> Observer:
        """Returns an Observer object"""

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

    Streams a translation dataset from Hugging Face Datasets, applies
    lightweight quality/length filters, collects at most
    `dataset_size * oversample` valid candidates (early-stopping once
    that target is reached), then deterministically subsamples exactly
    `dataset_size` examples using a seeded RNG.

    Attributes:
        _config (DatasetConfig): Dataset identity and filtering parameters
            (dataset, lang/config tag, split, dataset_size, oversample,
            tokens_min, tokens_max, ratio_max, lang_src, lang_tgt, seed).
        _dataset: Reserved for future use; not currently populated.
        _observer (TranslationDatasetBuilderObserver): Internal observer used to record
            counts/timings (seen, filtered, candidates, selected, etc.).
    """

    def __init__(self, config: DatasetConfig) -> None:
        """Initialize the builder.

        Args:
            config (DatasetConfig): Configuration specifying dataset identity,
                language keys, split, size target, oversample factor, minimum and
                maximum token thresholds (`tokens_min`, `tokens_max`), symmetric
                length-ratio constraint (`ratio_max`), and RNG seed.
        """
        self._config = config
        self._dataset = None
        self._observer = TranslationDatasetBuilderObserver(
            dataset=self._config.dataset,
            split=self._config.split,
            lang=self._config.lang,
            seed=self._config.seed,
            dataset_size=self._config.dataset_size,
            oversample=self._config.oversample,
        )

    @property
    def observer(self) -> TranslationDatasetBuilderObserver:
        """Observer: Internal observer with counters and timing."""
        return self._observer

    def build(self) -> Dataset:
        """Construct a `TranslationDataset`.

        Pipeline:
          1) Stream the configured split.
          2) Extract and filter until `dataset_size * oversample` valid candidates.
          3) Deterministically sample `dataset_size` examples with the configured seed.

        Returns:
            Dataset: A `TranslationDataset` containing exactly
            `config.dataset_size` `(src, tgt)` pairs.

        Raises:
            ValueError: If fewer than `config.dataset_size` valid rows are available
                after filtering.
        """
        with self._observer:
            stream = self._load_dataset()
            data = self._extract(data=stream)

        dataset = TranslationDataset.create(
            config=self._config, data=data, build_log=self._observer
        )
        return dataset

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

        Behavior:
        - Reads rows and normalizes each to a flat `{"src": ..., "tgt": ...}` view.
        - Applies `_is_valid_row` to enforce token-length and symmetric length-ratio
          constraints.
        - Collects at most `dataset_size * oversample` valid candidates (early stop).
        - Returns a deterministic sample of size `dataset_size` using `seed`.

        Args:
            data (IterableDataset): Streaming dataset iterator.

        Returns:
            List[Dict[str, Any]]: Exactly `config.dataset_size` dictionaries with
            keys `{"idx", "src", "tgt"}`.

        Raises:
            ValueError: If the number of valid candidates is less than
                `config.dataset_size`.
        """
        candidates: List[Dict[str, Any]] = []
        target = self._config.dataset_size * self._config.oversample

        for row in data:
            self._observer.seen += 1
            candidate = row.get("translation", row)
            if self._is_valid_row(candidate):
                self._observer.candidates += 1
                candidates.append(self._parse_row(candidate))
                if len(candidates) >= target:
                    break

        if len(candidates) < self._config.dataset_size:
            raise ValueError(
                f"Not enough valid rows: {len(candidates)} < {self._config.dataset_size}. "
                f"Increase oversample or relax filters."
            )

        rng = random.Random(self._config.seed)
        selected = rng.sample(candidates, self._config.dataset_size)
        self._observer.selected = len(selected)
        return selected

    def _is_valid_row(self, row: Dict[str, Any]) -> bool:
        """Validate a `(src, tgt)` pair against length and ratio constraints.

        Enforcement:
        - Both sides must be non-empty after whitespace split.
        - Token counts must satisfy `tokens_min` and `tokens_max` (per side).
        - Symmetric length ratio must satisfy `_is_valid_ratio` (≤ `ratio_max`).

        Args:
            row (Dict[str, Any]): A record containing source/target strings under
                the configured language keys (e.g., `"en"`, `"fr"`).

        Returns:
            bool: True if the row passes all filters; False otherwise.

        Raises:
            KeyError: If the expected language keys are missing in `row`
                (indicates a dataset/config mismatch).
        """
        try:
            src_len = len(row[self._config.lang_src].split())
            tgt_len = len(row[self._config.lang_tgt].split())
        except KeyError as e:
            present = list(row.keys())
            raise KeyError(
                f"Dataset/Config mismatch: missing key {e!s}. "
                f"Expected keys: {self._config.lang_src!r}, {self._config.lang_tgt!r}. "
                f"Row keys: {present}. "
                f"dataset={self._config.dataset} lang={self._config.lang} split={self._config.split}"
            )

        if src_len == 0 or tgt_len == 0:
            self._observer.filtered += 1
            self._observer.filtered_empty += 1
            return False

        if not self._is_valid_ratio(src_len=src_len, tgt_len=tgt_len):
            self._observer.filtered += 1
            self._observer.filtered_ratio += 1
            return False

        if src_len < self._config.tokens_min:
            self._observer.filtered += 1
            self._observer.filtered_src_short += 1
            return False

        if src_len > self._config.tokens_max:
            self._observer.filtered += 1
            self._observer.filtered_src_long += 1
            return False

        if tgt_len < self._config.tokens_min:
            self._observer.filtered += 1
            self._observer.filtered_tgt_short += 1
            return False

        if tgt_len > self._config.tokens_max:
            self._observer.filtered += 1
            self._observer.filtered_tgt_long += 1
            return False

        return True

    def _is_valid_ratio(self, src_len: int, tgt_len: int) -> bool:
        """Check the symmetric length ratio between source and target.

        Uses the symmetric ratio `max(src_len, tgt_len) / min(src_len, tgt_len)`
        and compares it to `config.ratio_max`.

        Args:
            src_len (int): Token count of the source text.
            tgt_len (int): Token count of the target text.

        Returns:
            bool: True if the symmetric ratio is within the allowed maximum; otherwise False.

        Notes:
            For asymmetric bounds, consider:
            `ratio_min <= (src_len / tgt_len) <= ratio_max`.
        """
        r = max(src_len, tgt_len) / min(src_len, tgt_len)
        return r <= self._config.ratio_max

    def _parse_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw row into a stable `(idx, src, tgt)` mapping.

        Computes `idx` as the first 12 hex chars of SHA1 over
        `{"src": src, "tgt": tgt}` to provide a compact, deterministic identifier.

        Args:
            row (Dict[str, Any]): Mapping with language keys containing the raw
                source and target strings.

        Returns:
            Dict[str, Any]: A dictionary with keys:
                - `idx` (str): Stable 12-hex identifier.
                - `src` (str): Source text (stripped).
                - `tgt` (str): Target text (stripped).
        """
        src = row[self._config.lang_src].strip()
        tgt = row[self._config.lang_tgt].strip()
        idx = sha1(repr({"src": src, "tgt": tgt}).encode("utf-8")).hexdigest()[:12]
        return {"idx": idx, "src": src, "tgt": tgt}
