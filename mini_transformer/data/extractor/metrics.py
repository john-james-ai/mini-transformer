#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/extractor/metrics.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 10:01:42 am                                                 #
# Modified   : Monday August 25th 2025 10:41:23 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from mini_transformer.data.base import MetricsCollector
from mini_transformer.utils.mixins import FreezableMixin, ObjectRepresentationMixin

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class ExtractorMetrics(MetricsCollector, FreezableMixin, ObjectRepresentationMixin):

    n: int = 0
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
        self.throughput = round(self.n / d, 3) if d > 0 else 0.0
        self.freeze()

    def log_summary(self) -> None:
        """Log a compact, human-readable summary at INFO level.

        Emits ``str(self)`` via the module logger. Customize ``__str__`` in
        the base class if you need a different format.
        """
        output = str(self)
        logger.debug(output)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TranslationDatasetExtractorMetrics(ExtractorMetrics):
    """Metrics collector for translation dataset extraction."""

    avg_seq_len: float = 0.0
    max_seq_len: int = 0
    min_seq_len: int = 0

    src_avg_seq_len: float = 0.0
    src_max_seq_len: int = 0
    src_min_seq_len: int = 0

    tgt_avg_seq_len: float = 0.0
    tgt_max_seq_len: int = 0
    tgt_min_seq_len: int = 0

    quant_data: Dict[str, List] = field(default_factory=dict)

    def add(self, key: str, value: Dict[str, Any]) -> None:
        if key not in self.quant_data:
            self.quant_data[key] = []
        self.quant_data[key].append(value)

    def get(self, key: str) -> List:
        return self.quant_data.get(key, [])

    def remove(self, key: str) -> None:
        if key in self.quant_data:
            del self.quant_data[key]

    def info(self) -> pd.DataFrame:
        items = []
        for key, value in self.quant_data.items():
            item = {"key": key, "rows": len(value)}
            items.append(item)
        return pd.DataFrame(items)

    def summarize(self) -> None:
        for key, values in self.quant_data.items():
            if values and isinstance(values[0], (int, float)):
                self.__setattr__(f"{key}_avg", sum(values) / len(values))
                self.__setattr__(f"{key}_max", max(values))
                self.__setattr__(f"{key}_min", min(values))
