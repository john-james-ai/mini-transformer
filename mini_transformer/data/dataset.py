#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/dataset.py                                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 11:17:40 pm                                                 #
# Modified   : Monday August 25th 2025 04:29:20 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Union

from mini_transformer.data.builder.metrics import TranslationDatasetBuilderMetrics
from mini_transformer.data.extractor.config import TranslationDatasetExtractorConfig
from mini_transformer.data.extractor.metrics import TranslationDatasetExtractorMetrics

if TYPE_CHECKING:
    from mini_transformer.data.base import MetricsCollector
    from mini_transformer.data.base import Config

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from mini_transformer.data.builder.config import TranslationDatasetBuilderConfig
from mini_transformer.utils.mixins import ObjectRepresentationMixin


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class Dataset(ABC, ObjectRepresentationMixin):
    # Metadata
    id: str  # Unique identifier for the dataset
    name: str  # Descriptive name for the dataset
    split: str  # train, validation, test
    n: int  # Number of rows in the dataset
    source: str  # Source ie HuggingFace
    source_dataset_name: str  # Dataset name at source
    stage: str  # Processing stage: raw, cleaned, tokenized, etc.

    # Dataset payload
    data: List[Dict[str, Any]]

    # Provenance
    created: datetime
    config: Union[TranslationDatasetBuilderConfig, TranslationDatasetExtractorConfig]
    metrics: Union[TranslationDatasetBuilderMetrics, TranslationDatasetExtractorMetrics]

    def __len__(self) -> int:
        """Number of examples in the dataset.

        Returns:
            int: Count of examples in `self.data`.
        """
        return len(self.data)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Iterate over examples.

        Yields:
            Dict[str, Any]: Each example from `self.data` in order.
        """
        return iter(self.data)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        """Indexed access to examples.

        Args:
            i: Zero-based index of the example to retrieve.

        Returns:
            Dict[str, Any]: The example at index `i`.

        Raises:
            IndexError: If `i` is out of range.
        """
        return self.data[i]

    # --- INFO ----------------------------------------------------------------
    @property
    @abstractmethod
    def info(self) -> Dict[str, Any]:
        """Returns select (immutable) information about the dataset in dictionary format."""

    # --- MATERIALIZE ---------------------------------------------------------
    def materialize(self, data: List[Dict[str, Any]]) -> Dataset:
        return replace(self, data=data)

    def dematerialize(self) -> Dataset:
        return replace(self, data=[])

    # --- IO ------------------------------------------------------------------
    def to_jsonl(self, path: str, *, ensure_ascii: bool = False) -> None:
        """Write examples to a JSON Lines file.

        Each element of `self.data` is serialized as one JSON object per line.

        Args:
            path: Output file path for the JSONL file.
            ensure_ascii: If True, non-ASCII characters are escaped in the
                output; if False (default), UTF-8 characters are preserved.

        Raises:
            OSError: If the file cannot be created or written.
            TypeError: If elements in `self.data` are not JSON-serializable.
        """
        filepath = Path(path)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for ex in self.data:
                f.write(json.dumps(ex, ensure_ascii=ensure_ascii) + "\n")

    def to_pandas(self) -> pd.DataFrame:
        """Render the dataset as a pandas DataFrame.

        Returns:
            pandas.DataFrame: A dataframe constructed from `self.data`. Column
                names are inferred from example keys (e.g., "idx", "src", "tgt").

        Notes:
            Requires pandas to be installed and importable as `pd`.
        """
        return pd.DataFrame.from_records(self.data)


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class TranslationDataset(Dataset):
    lang: str
    lang_src: str
    lang_tgt: str

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "split": self.split,
            "n": self.n,
            "lang": self.lang,
            "lang_src": self.lang_src,
            "lang_tgt": self.lang_tgt,
            "source": self.source,
            "source_dataset_name": self.source_dataset_name,
            "created": self.metrics.ended_at,
        }

    @classmethod
    def create(
        cls,
        config: Union[
            TranslationDatasetBuilderConfig, TranslationDatasetExtractorConfig
        ],
        metrics: Union[
            TranslationDatasetBuilderMetrics, TranslationDatasetExtractorMetrics
        ],
        data: List[Dict[str, Any]],
    ) -> TranslationDataset:
        """Creates a TranslationDataset from a TranslationDatasetBuilderConfig.

        Args:
            config: The configuration to create the Dataset from.
            metrics: The metrics collected during the build process.
            data: The dataset payload.

        Returns:
            TranslationDataset: The created Dataset.
        """
        return cls(
            id=config.fingerprint,
            name=config.name,
            split=config.split,
            n=metrics.n,
            stage=config.stage,
            lang=config.lang,
            lang_src=config.lang_src,
            lang_tgt=config.lang_tgt,
            source=config.source,
            source_dataset_name=config.source_dataset_name,
            data=data,
            created=datetime.now(),
            config=config,
            metrics=metrics,
        )
