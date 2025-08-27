#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/builder/extractor.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 10:00:42 am                                                 #
# Modified   : Tuesday August 26th 2025 11:57:27 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Dict, List, Optional

import pandas as pd
from dependency_injector.wiring import Provide, inject
from tqdm import tqdm

from mini_transformer.container import MiniTransformerContainer
from mini_transformer.data.builder.base import (
    Builder,
    DatasetBuilderConfig,
    MetricsCollector,
)
from mini_transformer.data.dataset import TranslationDataset
from mini_transformer.data.repo import DatasetRepo
from mini_transformer.infra.data_io.download import HFDatasetDownloader


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class TranslationDatasetExtractorBuilderConfig(DatasetBuilderConfig):
    """Configuration for extracting a raw translation dataset from a source.

    This dataclass holds the configuration for the initial extraction of a
    translation dataset, typically from a remote source like Hugging Face. It
    specifies the language pair and parameters for efficient, reproducible
    streaming and shuffling, which is useful for handling large datasets that
    may not fit into memory.

    Attributes:
        lang (str): The language pair identifier (e.g., "fr-en").
        lang_src (str): The source language code (e.g., "en").
        lang_tgt (str): The target language code (e.g., "fr").
        shuffle (bool): If True, the dataset is shuffled during extraction.
        buffer_size (int): The size of the buffer used for shuffling the
            dataset stream. A larger buffer results in better randomness.
        stage (str): The stage in the data processing lifecycle, ie 'raw'
        seed (int): The random seed for reproducible shuffling.
    """

    lang: str = field(default="fr-en", metadata={"stable": True})
    lang_src: str = field(default="en", metadata={"stable": True})
    lang_tgt: str = field(default="fr", metadata={"stable": True})
    shuffle: bool = field(default=True, metadata={"stable": True})
    buffer_size: int = field(default=30000, metadata={"stable": True})
    stage: bool = field(default="raw", metadata={"stable": True})
    seed: int = field(default=42, metadata={"stable": True})

    @property
    def dataset_name(self) -> str:
        """Generates a human-readable name for the dataset based on its configuration."""
        return f"{self.source_dataset_name}-{self.lang}-{self.split}-{self.stage}-{self.n}_examples-{self.fingerprint}"


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TranslationDatasetExtractorMetrics(MetricsCollector):
    """Collects detailed quantitative metrics during dataset extraction.

    This class extends `MetricsCollector` to compute statistics about the
    extracted dataset, such as min, max, and average sequence lengths for
    both the source and target text. It also provides a generic data store
    (`quant_data`) for collecting arbitrary data points for later analysis.

    Attributes:
        n (int): The total number of examples processed.
        throughput (float): The number of examples processed per second.
        avg_seq_len (float): The overall average sequence length.
        max_seq_len (int): The overall maximum sequence length.
        min_seq_len (int): The overall minimum sequence length.
        src_avg_seq_len (float): The average sequence length of the source text.
        src_max_seq_len (int): The maximum sequence length of the source text.
        src_min_seq_len (int): The minimum sequence length of the source text.
        tgt_avg_seq_len (float): The average sequence length of the target text.
        tgt_max_seq_len (int): The maximum sequence length of the target text.
        tgt_min_seq_len (int): The minimum sequence length of the target text.
        quant_data (Dict[str, List]): A generic store for collecting lists of
            quantitative data points under a specific key.
    """

    n: int = 0
    throughput: float = 0.0

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

    def end(self) -> None:
        """Finalizes metric collection, computes throughput, and freezes the object.

        This method calls the parent `end()` to stop the timer, calculates
        the final throughput based on the number of processed items, and then
        freezes the instance to prevent further changes.
        """
        super().end()
        d = self.duration
        self.throughput = round(self.n / d, 3) if d > 0 else 0.0
        self.freeze()

    def add(self, key: str, value: Dict[str, Any]) -> None:
        """Adds a data point to the quantitative data store.

        Appends a value to the list associated with the given key in the
        `quant_data` dictionary. If the key does not exist, it is created.

        Args:
            key: The name of the data series to add to.
            value: The data point (e.g., a dictionary) to append.
        """
        if key not in self.quant_data:
            self.quant_data[key] = []
        self.quant_data[key].append(value)

    def get(self, key: str) -> List:
        """Retrieves a data series from the quantitative data store.

        Args:
            key: The name of the data series to retrieve.

        Returns:
            The list of data points associated with the key, or an empty
            list if the key does not exist.
        """
        return self.quant_data.get(key, [])

    def remove(self, key: str) -> None:
        """Removes a data series from the quantitative data store.

        Args:
            key: The name of the data series to remove.
        """
        if key in self.quant_data:
            del self.quant_data[key]

    def info(self) -> pd.DataFrame:
        """Returns a summary of the collected quantitative data.

        Returns:
            A pandas DataFrame summarizing the contents of the `quant_data`
            store, with columns for the key and the number of rows (data
            points) collected for that key.
        """
        items = []
        for key, value in self.quant_data.items():
            item = {"key": key, "rows": len(value)}
            items.append(item)
        return pd.DataFrame(items)


# ------------------------------------------------------------------------------------------------ #
class TranslationDatasetExtractorBuilder(Builder):
    """Builds a raw translation dataset by extracting it from a source.

    This class orchestrates the initial download and quantitative analysis of a
    dataset. It uses a dedicated downloader component to stream data, then
    iterates through the downloaded examples to compute detailed statistics
    about sequence lengths (min, max, average) for the source, target, and
    combined texts. The final output is a `TranslationDataset` object that
    includes both the raw data and the computed metrics.
    """

    @inject
    def __init__(
        self,
        config: TranslationDatasetExtractorBuilderConfig,
        metrics: type[
            TranslationDatasetExtractorMetrics
        ] = TranslationDatasetExtractorMetrics,
        downloader: type[HFDatasetDownloader] = HFDatasetDownloader,
        repo: DatasetRepo = Provide[MiniTransformerContainer.data.repo],
    ) -> None:
        self._config = config
        self._downloader = downloader(
            dataset=config.source_dataset_name,
            language=config.lang,
            split=config.split,
            n=config.n,
            shuffle=config.shuffle,
            buffer_size=config.buffer_size,
            seed=config.seed,
        )

        self._repo = repo
        self._metrics = metrics()

    @property
    def metrics(self) -> Optional[TranslationDatasetExtractorMetrics]:
        """Provides access to the metrics collector for this build process.

        Returns:
            TranslationDatasetExtractorMetrics: The metrics object containing
            detailed quantitative statistics and timing data for the extraction.
        """
        return self._metrics

    def build(self) -> TranslationDataset:
        """Executes the download and statistical analysis to build the dataset.

        This method orchestrates the entire extraction process. It first
        downloads the raw data stream. It then iterates through each example,
        calculating sequence lengths based on word counts for the source and
        target texts. These statistics are aggregated and stored in the
        metrics object before being used to create the final, raw
        `TranslationDataset`.

        Returns:
            A new dataset instance containing the raw extracted data and a
            rich set of quantitative metrics.
        """
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
        data = []
        with self._metrics:
            dataset = self._downloader.download()
            for row in tqdm(dataset, total=self._config.n):
                row = row.get("translation", row)
                data.append(row)
                words_src = row[self._config.lang_src].split()
                words_tgt = row[self._config.lang_tgt].split()
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
                self._metrics.add("seq_len_data", row_data)

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

            self._metrics.n = metrics["n"]
            self._metrics.avg_seq_len = metrics["avg_seq_len"]
            self._metrics.max_seq_len = metrics["max_seq_len"]
            self._metrics.min_seq_len = metrics["min_seq_len"]
            self._metrics.src_avg_seq_len = metrics["src_avg_seq_len"]
            self._metrics.src_max_seq_len = metrics["src_max_seq_len"]
            self._metrics.src_min_seq_len = metrics["src_min_seq_len"]
            self._metrics.tgt_avg_seq_len = metrics["tgt_avg_seq_len"]
            self._metrics.tgt_max_seq_len = metrics["tgt_max_seq_len"]
            self._metrics.tgt_min_seq_len = metrics["tgt_min_seq_len"]

        dataset = TranslationDataset.create(
            config=self._config,
            metrics=self._metrics,
            data=data,
        )

        self._metrics.log_summary

        return dataset
