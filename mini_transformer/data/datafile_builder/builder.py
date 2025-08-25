#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/datafile_builder/builder.py                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday August 23rd 2025 12:02:51 am                                               #
# Modified   : Monday August 25th 2025 09:34:47 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import logging
import random
from hashlib import sha1
from typing import Any, Dict, List

from datasets import load_dataset
from dependency_injector.wiring import Provide, inject
from tqdm import tqdm

from mini_transformer.container import MiniTransformerContainer
from mini_transformer.data.base import Builder
from mini_transformer.data.datafile import TranslationDataFile
from mini_transformer.data.datafile_builder.config import (
    TranslationDataFileBuilderConfig,
)
from mini_transformer.data.datafile_builder.metrics import (
    TranslationDataFileBuilderMetrics,
)
from mini_transformer.infra.data_io.download import HFDatasetDownloader

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
class TranslationDataFileBuilder(Builder):

    @inject
    def __init__(
        self,
        builder_config: TranslationDataFileBuilderConfig,
        downloader: HFDatasetDownloader = Provide[
            MiniTransformerContainer.infra.hf_downloader
        ],
    ) -> None:
        self._downloader = downloader
        self._builder_config = builder_config
        self._builder_metrics = TranslationDataFileBuilderMetrics()

    @property
    def builder_metrics(self) -> TranslationDataFileBuilderMetrics:
        """Metrics: Internal builder_metrics with counters and timing."""
        return self._builder_metrics

    def build(self) -> None:
        n = 0
        src_len = []
        tgt_len = []
        with self._builder_metrics:
            data = self._downloader.download(
                split=self._builder_config.split, n=self._builder_config.n
            )
            for row in tqdm(data):
                row = row.get("translation", row)
                src_seq_len = len(row[self._builder_config.lang_src].split())
                tgt_seq_len = len(row[self._builder_config.lang_tgt].split())
                src_len.append(src_seq_len)
                tgt_len.append(tgt_seq_len)
                n += 1

        src_tgt_len = src_len + tgt_len
        self._builder_metrics.n = n
        self._builder_metrics.src_len_min = min(src_len)
        self._builder_metrics.src_len_max = max(src_len)
        self._builder_metrics.src_len_avg = (
            sum(src_len) / len(src_len) if src_len else 0
        )
        self._builder_metrics.tgt_len_min = min(tgt_len)
        self._builder_metrics.tgt_len_max = max(tgt_len)
        self._builder_metrics.tgt_len_avg = (
            sum(tgt_len) / len(tgt_len) if tgt_len else 0
        )
        self._builder_metrics.seq_len_min = min(src_tgt_len) if src_tgt_len else 0
        self._builder_metrics.seq_len_max = max(src_tgt_len) if src_tgt_len else 0
        self._builder_metrics.seq_len_avg = (
            sum(src_tgt_len) / len(src_tgt_len) if src_tgt_len else 0
        )

        dataset = TranslationDataFile.create(
            builder_config=self._builder_config,
            builder_metrics=self._builder_metrics,
            data=data,
        )
        return dataset

    def _load_dataset(self) -> IterableDataFile:
        return load_dataset(
            self._builder_config.source_dataset_name,
            self._builder_config.lang,
            split=self._builder_config.split,
            streaming=True,
            # type: ignore[reportArgumentType]
        )

    def _extract(self, data: IterableDataFile) -> List[Dict[str, Any]]:
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
                f"DataFile/Config mismatch: missing key {e!s}. "
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
