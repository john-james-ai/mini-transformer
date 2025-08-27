#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/builder/data_filter.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday August 23rd 2025 12:02:51 am                                               #
# Modified   : Wednesday August 27th 2025 12:40:50 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import logging
import math
import random
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Dict, List

from mini_transformer.data.builder.base import (
    Builder,
    DatasetBuilderConfig,
    MetricsCollector,
)
from mini_transformer.data.dataset import TranslationDataset

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class TranslationDatasetBuilderFilteredConfig(DatasetBuilderConfig):
    """Configuration for filtering and building a translation dataset.

    This dataclass holds all parameters for screening a large, raw translation
    corpus into a smaller, cleaner subset suitable for training. It defines
    criteria for filtering based on token length estimates, source-to-target
    length ratios, and controls the addition of special tokens (BOS/EOS) for
    the source, target, and label sequences.

    Attributes:
        lang (str): The language pair identifier (e.g., "fr-en").
        lang_src (str): The source language code (e.g., "en").
        lang_tgt (str): The target language code (e.g., "fr").
        tokens_min (int): The minimum number of tokens for a sequence.
        tokens_max (int): The maximum number of tokens for a sequence.
        ratio_min (float): The minimum allowed ratio of source length to
            target length to filter misaligned pairs.
        ratio_max (float): The maximum allowed ratio of source length to
            target length to filter misaligned pairs.
        src_bos (bool): If True, adds a BOS token to the source sequence.
        src_eos (bool): If True, adds an EOS token to the source sequence.
        tgt_bos (bool): If True, adds a BOS token to the target (decoder
            input) sequence.
        tgt_eos (bool): If True, adds an EOS token to the target (decoder
            input) sequence.
        lbl_bos (bool): If True, adds a BOS token to the final label sequence.
        lbl_eos (bool): If True, adds an EOS token to the final label sequence.
        stage (str): The stage in the data processing lifecycle, ie 'filtered'
        seed (int): The random seed for reproducibility in sampling.
    """

    lang: str = field(default="fr-en", metadata={"stable": True})
    lang_src: str = field(default="en", metadata={"stable": True})
    lang_tgt: str = field(default="fr", metadata={"stable": True})
    tokens_min: int = field(default=8, metadata={"stable": True})
    tokens_max: int = field(default=64, metadata={"stable": True})
    ratio_min: float = field(default=0.5, metadata={"stable": True})
    ratio_max: float = field(default=2.0, metadata={"stable": True})
    src_bos: bool = field(default=False, metadata={"stable": True})
    src_eos: bool = field(default=False, metadata={"stable": True})
    tgt_bos: bool = field(default=True, metadata={"stable": True})
    tgt_eos: bool = field(default=False, metadata={"stable": True})
    lbl_bos: bool = field(default=False, metadata={"stable": True})
    lbl_eos: bool = field(default=True, metadata={"stable": True})
    stage: bool = field(default="filtered", metadata={"stable": True})
    seed: int = field(default=42, metadata={"stable": True})

    @property
    def dataset_name(self) -> str:
        """Name: Unique name for the dataset."""
        return f"{self.source_dataset_name}-{self.lang}-{self.split}-{self.stage}-{self.n}_examples-{self.fingerprint}"

    @property
    def src_max_words(self) -> int:
        """Compute the source-side word cap implied by ``tokens_max``.

        Returns:
            int: Word-count cap for source strings.
        """
        return self._compute_word_cap_from_tokens(bos=self.src_bos, eos=self.src_eos)

    @property
    def tgt_max_words(self) -> int:
        """Compute the target-side word cap implied by ``tokens_max``.

        Returns:
            int: Word-count cap for target strings.
        """
        return self._compute_word_cap_from_tokens(bos=self.tgt_bos, eos=self.tgt_eos)

    def _compute_word_cap_from_tokens(
        self, bos: bool, eos: bool, r: float = 1.3, margin: int = 8
    ) -> int:
        """Derive a word-count cap that rarely exceeds ``tokens_max`` after tokenization.

        This converts the model's token budget into a conservative word cap using
        a tokens-per-word estimate and a small margin for punctuation/edge cases.
        It lets you filter long strings *before* tokenization while keeping the
        vast majority of examples under the model's ``tokens_max``.

        Args:
            bos (bool): Whether a BOS token is added for this side.
            eos (bool): Whether an EOS token is added for this side.
            r (float): Estimated tokens per word (EN/FR with unigram SPM and
                smallish vocab ~ ``1.5``; use ``1.7`` for very small vocabs,
                ``1.3`` for larger).
            margin (int): Token headroom reserved for punctuation/oddities.

        Returns:
            int: Word-count cap to apply on raw text during extraction.
        """
        specials = int(bos) + int(eos)
        budget = self.tokens_max - specials - margin
        return max(1, math.floor(budget / r))


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TranslationDatasetBuilderFilteredMetrics(MetricsCollector):
    """Collects detailed metrics for the translation dataset filtering process.

    This class extends `MetricsCollector` with specific counters to track the
    data filtering and selection funnel. It provides a comprehensive summary of
    how many source examples were seen, how many were filtered out for
    specific reasons (e.g., length, ratio), and how many were ultimately
    selected for the final dataset.

    Attributes:
        n (int): The target number of examples to select for the dataset.
        seen (int): The total number of examples read from the source dataset.
        filtered (int): The total number of examples discarded for any reason.
        candidates (int): The number of examples that passed all filters and
            are eligible for selection.
        selected (int): The final number of examples selected for the dataset.
        filtered_empty (int): Count of examples filtered because the source or
            target text was empty.
        filtered_ratio (int): Count of examples filtered due to an invalid
            source-to-target length ratio.
        filtered_src_short (int): Count of examples filtered because the source
            text was too short.
        filtered_src_long (int): Count of examples filtered because the source
            text was too long.
        filtered_tgt_short (int): Count of examples filtered because the target
            text was too short.
        filtered_tgt_long (int): Count of examples filtered because the target
            text was too long.
        throughput (float): The number of examples seen per second, calculated
            as `seen / duration`.
    """

    n: int = 0

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

    def end(self) -> None:
        """Finalize metrics, compute throughput, and freeze the snapshot.

        Calls :meth:`super().end()` to close timing and set ``duration``,
        computes ``throughput = round(seen / duration, 3)`` when duration
        is positive (else 0.0), and invokes :meth:`freeze` to prevent
        further mutation.
        """
        super().end()
        d = self.duration
        self.throughput = round(self.seen / d, 3) if d > 0 else 0.0
        self.freeze()


# ------------------------------------------------------------------------------------------------ #
class TranslationDatasetBuilderFiltered(Builder):
    """Builds a filtered translation dataset from a raw data source.

    This class implements the Builder pattern to process a raw
    `TranslationDataset`. It shuffles the source data and then iterates through
    it, applying a series of validation checks based on the provided
    `TranslationDatasetBuilderFilteredConfig`.

    The filtering criteria include checking for empty strings, validating the
    source-to-target length ratio, and ensuring sequence lengths fall within a
    specified min/max range. The process stops once the target number of
    valid examples has been collected. The final output is a new, clean
    `TranslationDataset` containing the selected data and detailed metrics
    about the filtering process.
    """

    def __init__(
        self,
        dataset: TranslationDataset,
        config: TranslationDatasetBuilderFilteredConfig,
        metrics: type[
            TranslationDatasetBuilderFilteredMetrics
        ] = TranslationDatasetBuilderFilteredMetrics,
    ) -> None:
        self._data = dataset.data
        self._config = config
        self._metrics = metrics()

    @property
    def metrics(self) -> TranslationDatasetBuilderFilteredMetrics:
        """Provides access to the metrics collector for this build process.

        Returns:
            TranslationDatasetBuilderFilteredMetrics: The metrics object
            containing counters and timing data for the build.
        """
        return self._metrics

    def build(self) -> TranslationDataset:
        """Executes the filtering and sampling process to build the dataset.

        This method shuffles the source data, iterates through it applying
        validation checks, and collects the first `n` valid examples as
        specified in the config. The associated metrics object is populated
        during the run.

        Returns:
            TranslationDataset: A new, filtered dataset instance.
        """
        candidates: List[Dict[str, Any]] = []

        with self._metrics:
            random.seed(self._config.seed)
            random.shuffle(self._data)

            for row in self._data:
                self._metrics.seen += 1
                candidate = row.get("translation", row)
                if self._is_valid_row(candidate):
                    self._metrics.n += 1
                    candidates.append(self._parse_row(candidate))
                    if len(candidates) >= self._config.n:
                        break

            self._metrics.selected = len(candidates)

            dataset = TranslationDataset.create(
                config=self._config,
                metrics=self._metrics,
                data=candidates,
            )

        return dataset

    def _is_valid_row(self, row: Dict[str, Any]) -> bool:
        """Checks if a single data row meets all filtering criteria.

        This method applies all validation rules from the configuration, such
        as checking for empty strings, length ratio, and min/max word counts.
        It increments the relevant `filtered_*` counters in the metrics object
        if a check fails.

        Args:
            row: A dictionary representing a single example from the source
                dataset.

        Returns:
            True if the row is valid, False otherwise.

        Raises:
            KeyError: If the language keys specified in the config are not
                present in the row.
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
                f"dataset={self._config.source_dataset_name} lang={self._config.lang} split={self._config.split}"
            )

        if src_len == 0 or tgt_len == 0:
            self._metrics.filtered += 1
            self._metrics.filtered_empty += 1
            return False

        if not self._is_valid_ratio(src_len=src_len, tgt_len=tgt_len):
            self._metrics.filtered += 1
            self._metrics.filtered_ratio += 1
            return False

        if src_len < self._config.tokens_min:
            self._metrics.filtered += 1
            self._metrics.filtered_src_short += 1
            return False

        if src_len > self._config.tokens_max:
            self._metrics.filtered += 1
            self._metrics.filtered_src_long += 1
            return False

        if tgt_len < self._config.tokens_min:
            self._metrics.filtered += 1
            self._metrics.filtered_tgt_short += 1
            return False

        if tgt_len > self._config.tokens_max:
            self._metrics.filtered += 1
            self._metrics.filtered_tgt_long += 1
            return False

        return True

    def _is_valid_ratio(self, src_len: int, tgt_len: int) -> bool:
        """Validates the ratio of source length to target length.

        Args:
            src_len: The length of the source text in words.
            tgt_len: The length of the target text in words.

        Returns:
            True if the ratio is within the configured limits, False otherwise.
        """
        if src_len == 0 or tgt_len == 0:
            return False  # Avoid division by zero
        r = max(src_len, tgt_len) / min(src_len, tgt_len)
        return r <= self._config.ratio_max

    def _parse_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Strips, formats, and adds a unique ID to a valid data row.

        This method takes a raw candidate row, strips leading/trailing
        whitespace from the source and target strings, and generates a unique
        content-based ID from a hash of the cleaned text.

        Args:
            row: The valid candidate row to be processed.

        Returns:
            A cleaned dictionary with 'idx', 'src', and 'tgt' keys.
        """
        src = row[self._config.lang_src].strip()
        tgt = row[self._config.lang_tgt].strip()
        idx = sha1(repr({"src": src, "tgt": tgt}).encode("utf-8")).hexdigest()[:12]
        return {"idx": idx, "src": src, "tgt": tgt}
