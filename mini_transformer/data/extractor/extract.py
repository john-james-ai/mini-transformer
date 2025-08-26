#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/extractor/extract.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 10:00:42 am                                                 #
# Modified   : Monday August 25th 2025 11:11:11 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import json
from abc import ABC, abstractmethod
from hashlib import sha1
from typing import Optional

from dependency_injector.wiring import Provide, inject
from tqdm import tqdm

from mini_transformer.container import MiniTransformerContainer
from mini_transformer.data.dataset import TranslationDataset
from mini_transformer.data.extractor.config import TranslationDatasetExtractorConfig
from mini_transformer.data.extractor.metrics import (
    ExtractorMetrics,
    TranslationDatasetExtractorMetrics,
)
from mini_transformer.data.repo import DatasetRepo
from mini_transformer.infra.data_io.download import HFDatasetDownloader


# ------------------------------------------------------------------------------------------------ #
class Extractor(ABC):
    """Abstract base class for all extractors."""

    @property
    @abstractmethod
    def extractor_metrics(self) -> ExtractorMetrics:
        """Metrics: Internal extractor_metrics with counters and timing."""
        pass

    @abstractmethod
    def build(self) -> None:
        """Builds the object."""
        pass


# ------------------------------------------------------------------------------------------------ #
class TranslationDatasetExtractor:

    @inject
    def __init__(
        self,
        extractor_config: TranslationDatasetExtractorConfig,
        extractor_metrics: type[
            TranslationDatasetExtractorMetrics
        ] = TranslationDatasetExtractorMetrics,
        downloader: type[HFDatasetDownloader] = HFDatasetDownloader,
        repo: DatasetRepo = Provide[MiniTransformerContainer.data.repo],
    ) -> None:
        self._extractor_config = extractor_config
        self._downloader = downloader(
            dataset=extractor_config.source_dataset_name,
            language=extractor_config.lang,
            split=extractor_config.split,
            n=extractor_config.n,
            shuffle=extractor_config.shuffle,
            buffer_size=extractor_config.buffer_size,
            seed=extractor_config.seed,
        )

        self._repo = repo
        self._extractor_metrics = extractor_metrics()

    @property
    def extractor_metrics(self) -> Optional[TranslationDatasetExtractorMetrics]:
        """Metrics: Internal extractor_metrics with counters and timing."""
        return self._extractor_metrics

    def extract(self) -> TranslationDataset:
        n = 0
        src_len = []
        tgt_len = []

        metrics = {
            "n": 0,
            "avg_seq_len": 0.0,
            "max_seq_len": 0,
            "min_seq_len": 0,
            "src_avg_seq_len": 0.0,
            "src_max_seq_len": 0,
            "src_min_seq_len": 0,
            "tgt_avg_seq_len": 0.0,
            "tgt_max_seq_len": 0,
            "tgt_min_seq_len": 0,
        }

        seq_len_data = dict()

        with self._extractor_metrics:
            data = self._downloader.download()
            for row in tqdm(data, total=self._extractor_config.n):
                row = row.get("translation", row)
                words_src = row[self._extractor_config.lang_src].split()
                words_tgt = row[self._extractor_config.lang_tgt].split()
                words = words_src + words_tgt

                src_seq_len = len(words_src)
                tgt_seq_len = len(words_tgt)
                max_len = max(src_seq_len, tgt_seq_len + 1)

                src_len.append(src_seq_len)
                tgt_len.append(tgt_seq_len)
                n += 1

                s = json.dumps(words, sort_keys=True, separators=(",", ":"))
                row_data = {
                    "id": sha1(s.encode("utf-8")).hexdigest()[:8],
                    "src_len": src_seq_len,
                    "tgt_len": tgt_seq_len,
                    "max_len": max_len,
                }
                self._extractor_metrics.add("seq_len_data", row_data)

            src_tgt_len = src_len + tgt_len
            metrics["n"] = n
            metrics["src_min_seq_len"] = min(src_len)
            metrics["src_max_seq_len"] = max(src_len)
            metrics["src_avg_seq_len"] = (
                round(sum(src_len) / len(src_len), 3) if src_len else 0
            )
            metrics["tgt_min_seq_len"] = min(tgt_len)
            metrics["tgt_max_seq_len"] = max(tgt_len)
            metrics["tgt_avg_seq_len"] = (
                round(sum(tgt_len) / len(tgt_len), 3) if tgt_len else 0
            )
            metrics["min_seq_len"] = min(src_tgt_len) if src_tgt_len else 0
            metrics["max_seq_len"] = max(src_tgt_len) if src_tgt_len else 0
            metrics["avg_seq_len"] = (
                round(sum(src_tgt_len) / len(src_tgt_len), 3) if src_tgt_len else 0
            )

            self._extractor_metrics.n = metrics["n"]
            self._extractor_metrics.avg_seq_len = metrics["avg_seq_len"]
            self._extractor_metrics.max_seq_len = metrics["max_seq_len"]
            self._extractor_metrics.min_seq_len = metrics["min_seq_len"]
            self._extractor_metrics.src_avg_seq_len = metrics["src_avg_seq_len"]
            self._extractor_metrics.src_max_seq_len = metrics["src_max_seq_len"]
            self._extractor_metrics.src_min_seq_len = metrics["src_min_seq_len"]
            self._extractor_metrics.tgt_avg_seq_len = metrics["tgt_avg_seq_len"]
            self._extractor_metrics.tgt_max_seq_len = metrics["tgt_max_seq_len"]
            self._extractor_metrics.tgt_min_seq_len = metrics["tgt_min_seq_len"]

        dataset = TranslationDataset.create(
            config=self._extractor_config,
            metrics=self._extractor_metrics,
            data=data,
        )

        self._extractor_metrics.log_summary

        return dataset
