#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/builder/build.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Saturday August 23rd 2025 12:02:51 am                                               #
# Modified   : Monday August 25th 2025 09:54:36 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import logging
import random
from abc import ABC, abstractmethod
from hashlib import sha1
from typing import Any, Dict, List
from dependency_injector.wiring import Provide, inject  

from mini_transformer.container import MiniTransformerContainer
from mini_transformer.data.base import MetricsCollector
from mini_transformer.data.builder.config import TranslationDatasetBuilderConfig
from mini_transformer.data.builder.metrics import TranslationDatasetBuilderMetrics
from mini_transformer.data.dataset import Dataset, TranslationDataset
from mini_transformer.data.tokenize.bpe import BPETokenization

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
class Builder(ABC):
    """Abstract base class for all builders."""

    @property
    @abstractmethod
    def builder_metrics(self) -> MetricsCollector:
        """Metrics: Internal builder_metrics with counters and timing."""
        pass

    @abstractmethod
    def build(self) -> Dataset:
        """Builds the object."""
        pass


# ------------------------------------------------------------------------------------------------ #
class TranslationDatasetBuilder(Builder):

    @inject
    def __init__(
        self,
        dataset: TranslationDataset,
        builder_config: TranslationDatasetBuilderConfig,
        builder_metrics: type[
            TranslationDatasetBuilderMetrics
        ] = TranslationDatasetBuilderMetrics,
        tokenization: BPETokenization = Provide[MiniTransformerContainer.data.tokenization],
    ) -> None:
        self._data = dataset.data
        self._builder_config = builder_config
        self._builder_metrics = builder_metrics()
        self._tokenization = tokenization

    @property
    def builder_metrics(self) -> TranslationDatasetBuilderMetrics:
        """Metrics: Internal builder_metrics with counters and timing."""
        return self._builder_metrics

    def _select_examples(self) -> List[Dict[str, Any]]:
        
        candidates: List[Dict[str, Any]] = []
        
        with self._builder_metrics:            
            # Shuffle the data with seed to ensure randomness and reproducibility when sampling
            random.seed(self._builder_config.seed)
            random.shuffle(self._data)

            for row in self._data:
                self._builder_metrics.seen += 1
                candidate = row.get("translation", row)
                if self._is_valid_row(candidate):
                    self._builder_metrics.candidates += 1
                    candidates.append(self._parse_row(candidate))
                    if len(candidates) >= self._builder_config.n:
                        break

            self._builder_metrics.n = len(candidates)
            
        return candidates
    
    def _tokenize_examples(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        
        
    def _create_dataset(self, tokens: List[Dict[str, Any]]) -> Dataset:

            dataset = TranslationDataset.create(
                config=self._builder_config,
                metrics=self._builder_metrics,
                data=candidates,
            )

        return dataset

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
