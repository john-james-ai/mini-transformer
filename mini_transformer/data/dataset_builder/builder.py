#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/dataset_builder/builder.py                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday August 23rd 2025 12:02:51 am                                               #
# Modified   : Monday August 25th 2025 08:14:11 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import logging
import random
from hashlib import sha1
from typing import Any, Dict, List

from datasets import IterableDataset, load_dataset

from mini_transformer.data.base import Builder
from mini_transformer.data.datafile import TranslationDataFile
from mini_transformer.data.dataset import TranslationDataset
from mini_transformer.data.dataset_builder.config import TranslationDatasetBuilderConfig
from mini_transformer.data.dataset_builder.metrics import (
    TranslationDatasetBuilderMetrics,
)

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
class TranslationDatasetBuilder(Builder):

    def __init__(
        self,
        data_file: TranslationDataFile,
        builder_config: TranslationDatasetBuilderConfig,
    ) -> None:
        self._data_file = data_file
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
            data = self._extract(data=self._data_file.data)

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
