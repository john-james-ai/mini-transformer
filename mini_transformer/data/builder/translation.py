#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /mini_transformer/data/builder/translation.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday August 23rd 2025 12:02:51 am                                               #
# Modified   : Saturday August 23rd 2025 12:22:57 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import logging
import random
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Dict, List

from datasets import IterableDataset, load_dataset

from mini_transformer.data.builder.base import (
    DatasetBuilder,
    DatasetBuilderConfig,
    DatasetBuilderMetrics,
)
from mini_transformer.data.dataset.translation import TranslationDataset
from mini_transformer.utils.mixins import FingerprintMixin, ObjectRepresentationMixin

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class TranslationDatasetBuilderConfig(
    DatasetBuilderConfig, FingerprintMixin, ObjectRepresentationMixin
):
    """Configuration for a reproducible translation dataset sample.

    This dataclass defines the configuration for constructing a filtered
    and reproducible dataset, including source metadata, language settings,
    split selection, sampling size, filtering criteria, and random seed.
    Fields marked with ``metadata={"stable": True}`` participate in the
    stable fingerprint computed by ``FingerprintMixin.fingerprint()`` and
    are used to derive a deterministic manifest filename.

    Attributes:
        source (str): Dataset source identifier (e.g., "HuggingFace").
        source_dataset_name (str): Name of the dataset at the source
            (e.g., "wmt14").
        source_dataset_url (str): URL to the dataset resource.
        lang (str): Language pair/config tag (e.g., "fr-en").
        lang_src (str): Source language key in records (e.g., "en").
        lang_tgt (str): Target language key in records (e.g., "fr").
        split (str): Dataset split to use (e.g., "train").
        dataset_target_size (int): Desired number of examples in the
            processed dataset.
        oversample (int): Oversampling multiplier applied to the dataset
            to achieve target size.
        tokens_min (int): Minimum token length for examples to be retained.
        tokens_max (int): Maximum token length for examples to be retained.
        ratio_min (float): Minimum acceptable length ratio
            (source/target tokens).
        ratio_max (float): Maximum acceptable length ratio
            (source/target tokens).
        src_bos (bool): Whether to prepend a beginning-of-sequence (BOS)
            token to source sequences.
        src_eos (bool): Whether to append an end-of-sequence (EOS) token
            to source sequences.
        tgt_bos (bool): Whether to prepend a BOS token to target sequences.
        tgt_eos (bool): Whether to append an EOS token to target sequences.
        seed (int): Random seed for reproducibility of sampling and
            filtering.
    """

    source: str = field(default="HuggingFace", metadata={"stable": True})
    source_dataset_name: str = field(default="wmt14", metadata={"stable": True})
    source_dataset_url: str = field(
        default="https://huggingface.co/datasets/wmt/wmt14", metadata={"stable": True}
    )
    lang: str = field(default="fr-en", metadata={"stable": True})
    lang_src: str = field(default="en", metadata={"stable": True})
    lang_tgt: str = field(default="fr", metadata={"stable": True})
    split: str = field(default="train", metadata={"stable": True})
    dataset_target_size: int = field(default=120, metadata={"stable": True})
    oversample: int = field(default=10, metadata={"stable": True})
    tokens_min: int = field(default=8, metadata={"stable": True})
    tokens_max: int = field(default=256, metadata={"stable": True})
    ratio_min: float = field(default=0.5, metadata={"stable": True})
    ratio_max: float = field(default=2.0, metadata={"stable": True})
    src_bos: bool = field(default=False, metadata={"stable": True})
    src_eos: bool = field(default=False, metadata={"stable": True})
    tgt_bos: bool = field(default=True, metadata={"stable": True})
    tgt_eos: bool = field(default=True, metadata={"stable": True})
    seed: int = field(default=42, metadata={"stable": True})

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
        self, bos: bool, eos: bool, r: float = 1.5, margin: int = 8
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
        return max(1, budget // int(round(r)))


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TranslationDatasetBuilderMetrics(DatasetBuilderMetrics):
    """Build-time metrics for translation dataset construction.

    Captures both the **inputs** that parameterize a build (dataset name,
    split, language, seed, size, oversample) and the **counters** produced
    while scanning, filtering, and selecting examples. Designed to be
    populated during a single builder run and finalized via :meth:`end`.

    Attributes:
        source_dataset_name (str | None): Human-readable dataset name at
            the source (e.g., "wmt14"). Stored for provenance in reports.
        split (str | None): Source split processed (e.g., "train", "valid").
        lang (str | None): Language pair tag (e.g., "en-fr").
        seed (int | None): PRNG seed used during sampling/selection.
        dataset_target_size (int | None): Desired number of items in the
            final artifact after selection.
        oversample (int | None): Oversampling multiplier applied to
            increase candidate pool before selection.

        seen (int): Total examples inspected by the builder (pre-filter).
        filtered (int): Total examples removed by any filter (should equal
            the sum of per-reason filtered_* counters).
        candidates (int): Examples surviving initial validation and
            eligible for final selection.
        selected (int): Examples chosen for the final dataset.

        filtered_empty (int): Removed because either source or target is
            empty/whitespace-only after normalization.
        filtered_ratio (int): Removed because src/tgt length ratio fell
            outside configured bounds.
        filtered_src_short (int): Removed due to source length < tokens_min.
        filtered_src_long (int): Removed due to source length > tokens_max.
        filtered_tgt_short (int): Removed due to target length < tokens_min.
        filtered_tgt_long (int): Removed due to target length > tokens_max.

        throughput (float): Processing throughput in examples/second,
            computed at :meth:`end` as ``seen / duration`` and rounded to
            three decimals.

    Notes:
        - ``duration`` is provided by the base class after calling
          :meth:`super().end()`.
        - :meth:`freeze` is expected to make the metrics immutable or mark
          them as finalized, depending on the base implementation.
    """

    source_dataset_name: str = None
    split: str = None
    lang: str = None
    seed: int = None
    dataset_target_size: int = None
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

    def log_summary(self) -> None:
        """Log a compact, human-readable summary at INFO level.

        Emits ``str(self)`` via the module logger. Customize ``__str__`` in
        the base class if you need a different format.
        """
        output = str(self)
        logger.info(output)


# ------------------------------------------------------------------------------------------------ #
class TranslationDatasetBuilder(DatasetBuilder):

    def __init__(self, builder_config: TranslationDatasetBuilderConfig) -> None:

        self._builder_config = builder_config
        self._dataset = None
        self._builder_metrics = TranslationDatasetBuilderMetrics(
            source_dataset_name=self._builder_config.source_dataset_name,
            split=self._builder_config.split,
            lang=self._builder_config.lang,
            seed=self._builder_config.seed,
            dataset_target_size=self._builder_config.dataset_target_size,
            oversample=self._builder_config.oversample,
        )

    @property
    def builder_metrics(self) -> TranslationDatasetBuilderMetrics:
        """Metrics: Internal builder_metrics with counters and timing."""
        return self._builder_metrics

    def build(self) -> TranslationDataset:
        with self._builder_metrics:
            stream = self._load_dataset()
            data = self._extract(data=stream)

        dataset = TranslationDataset.create(
            builder_config=self._builder_config,
            builder_metrics=self._builder_metrics,
            data=data,
        )
        return dataset

    def _load_dataset(self) -> IterableDataset:
        return load_dataset(
            self._builder_config.source_dataset_name,
            self._builder_config.lang,
            split=self._builder_config.split,
            streaming=True,
            # type: ignore[reportArgumentType]
        )

    def _extract(self, data: IterableDataset) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []
        target = (
            self._builder_config.dataset_target_size * self._builder_config.oversample
        )

        for row in data:
            self._builder_metrics.seen += 1
            candidate = row.get("translation", row)
            if self._is_valid_row(candidate):
                self._builder_metrics.candidates += 1
                candidates.append(self._parse_row(candidate))
                if len(candidates) >= target:
                    break

        if len(candidates) < self._builder_config.dataset_target_size:
            raise ValueError(
                f"Not enough valid rows: {len(candidates)} < {self._builder_config.dataset_target_size}. "
                f"Increase oversample or relax filters."
            )

        rng = random.Random(self._builder_config.seed)
        selected = rng.sample(candidates, self._builder_config.dataset_target_size)
        self._builder_metrics.selected = len(selected)
        return selected

    def _is_valid_row(self, row: Dict[str, Any]) -> bool:
        try:
            src_len = len(row[self._builder_config.lang_src].split())
            tgt_len = len(row[self._builder_config.lang_tgt].split())
        except KeyError as e:
            present = list(row.keys())
            raise KeyError(
                f"Dataset/Config mismatch: missing key {e!s}. "
                f"Expected keys: {self._builder_config.lang_src!r}, {self._builder_config.lang_tgt!r}. "
                f"Row keys: {present}. "
                f"dataset={self._builder_config.source_dataset_name} lang={self._builder_config.lang} split={self._builder_config.split}"
            )

        if src_len == 0 or tgt_len == 0:
            self._builder_metrics.filtered += 1
            self._builder_metrics.filtered_empty += 1
            return False

        if not self._is_valid_ratio(src_len=src_len, tgt_len=tgt_len):
            self._builder_metrics.filtered += 1
            self._builder_metrics.filtered_ratio += 1
            return False

        if src_len < self._builder_config.tokens_min:
            self._builder_metrics.filtered += 1
            self._builder_metrics.filtered_src_short += 1
            return False

        if src_len > self._builder_config.tokens_max:
            self._builder_metrics.filtered += 1
            self._builder_metrics.filtered_src_long += 1
            return False

        if tgt_len < self._builder_config.tokens_min:
            self._builder_metrics.filtered += 1
            self._builder_metrics.filtered_tgt_short += 1
            return False

        if tgt_len > self._builder_config.tokens_max:
            self._builder_metrics.filtered += 1
            self._builder_metrics.filtered_tgt_long += 1
            return False

        return True

    def _is_valid_ratio(self, src_len: int, tgt_len: int) -> bool:
        r = max(src_len, tgt_len) / min(src_len, tgt_len)
        return r <= self._builder_config.ratio_max

    def _parse_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        src = row[self._builder_config.lang_src].strip()
        tgt = row[self._builder_config.lang_tgt].strip()
        idx = sha1(repr({"src": src, "tgt": tgt}).encode("utf-8")).hexdigest()[:12]
        return {"idx": idx, "src": src, "tgt": tgt}
