#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /mini_transformer/data/dataset/base.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 11:17:40 pm                                                 #
# Modified   : Saturday August 23rd 2025 12:41:38 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List

import pandas as pd

if TYPE_CHECKING:
    from mini_transformer.data.builder.base import (
        DatasetBuilderConfig,
        DatasetBuilderMetrics,
    )

from mini_transformer.utils.mixins import ObjectRepresentationMixin


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class Dataset(ABC, ObjectRepresentationMixin):
    # Metadata
    id: str  # Unique identifier for the dataset
    name: str  # Human readable identifying name
    split: str  # train, validation, test
    size: int  # Number of rows in the dataset
    source: str  # Source ie HuggingFace
    source_dataset_name: str  # Dataset name at source

    # Dataset payload
    data: List[Dict[str, Any]]

    # Provenance
    created: datetime
    builder_config: DatasetBuilderConfig
    builder_metrics: DatasetBuilderMetrics

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
