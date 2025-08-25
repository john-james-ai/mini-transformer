#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/dataset_builder/metrics.py                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 11:57:08 pm                                                 #
# Modified   : Monday August 25th 2025 08:15:59 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
from dataclasses import dataclass

from mini_transformer.data.base import BuilderMetrics

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TranslationDatasetBuilderMetrics(BuilderMetrics):
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
