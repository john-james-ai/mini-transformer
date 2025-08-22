#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /dataset.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 08:17:31 am                                                #
# Modified   : Friday August 22nd 2025 05:43:53 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import json
from abc import ABC
from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Dict, List

import pandas as pd

from mini_transformer.data.config import DatasetConfig

if TYPE_CHECKING:
    from mini_transformer.data.builder import TranslationDatasetBuilderObserver


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class Dataset(ABC):
    """Immutable container for a dataset and its build configuration.

    This base class holds a stable identifier, the (frozen) configuration
    used to produce the data, and the materialized examples. It provides
    simple iteration, indexing, and JSONL export; subclasses may add
    format-specific helpers (e.g., `to_pandas()`).

    Attributes:
        dataset_id: Stable identifier for this dataset instance, typically
            derived from a fingerprint of the build configuration.
        dataset_name: A human-readable name for the dataset.
        config: The frozen `DatasetConfig` used to build this dataset.
        data: Materialized examples as a list of dictionaries. The schema is
            defined by the concrete dataset type (e.g., TranslationDataset
            expects keys like {"idx", "src", "tgt"}).
    """

    dataset_id: str
    dataset_name: str
    config: DatasetConfig
    data: List[Dict[str, Any]]

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

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.dataset_name,
            "size": len(self),
        }

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

        with open(path, "w", encoding="utf-8") as f:
            for ex in self.data:
                f.write(json.dumps(ex, ensure_ascii=ensure_ascii) + "\n")


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class TranslationDataset(Dataset):
    """Immutable container for parallel text examples.

    This concrete dataset represents translation pairs. Each example in
    `data` is expected to be a mapping with at least:
    - `idx` (str): Stable identifier for the pair.
    - `src` (str): Source text.
    - `tgt` (str): Target text.

    Attributes:
        build_log: A frozen `TranslationDatasetBuilderObserver` instance
            (or equivalent) captured at build time for provenance and timing.
    """

    build_log: TranslationDatasetBuilderObserver

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.dataset_name,
            "size": len(self),
            "created": self.build_log.ended_at,
        }

    # --- IO ------------------------------------------------------------------
    def to_pandas(self) -> pd.DataFrame:
        """Render the dataset as a pandas DataFrame.

        Returns:
            pandas.DataFrame: A dataframe constructed from `self.data`. Column
                names are inferred from example keys (e.g., "idx", "src", "tgt").

        Notes:
            Requires pandas to be installed and importable as `pd`.
        """
        return pd.DataFrame.from_records(self.data)

    @classmethod
    def create(
        cls,
        config: DatasetConfig,
        build_log: TranslationDatasetBuilderObserver,
        data: List[Dict[str, Any]],
    ) -> TranslationDataset:
        """Construct a `TranslationDataset` from builder outputs.

        This helper assigns `dataset_id` from the configuration fingerprint,
        attaches the frozen `build_log`, and stores the provided `data`
        unchanged.

        Args:
            config: The frozen `DatasetConfig` used to produce `data`. Must
                expose a `fingerprint` accessor (property or method) that
                yields a stable identifier.
            build_log: The finalized, frozen observer produced by the builder,
                containing counts and timing for this build.
            data: Materialized translation pairs as a list of dictionaries
                (each with keys like {"idx", "src", "tgt"}).

        Returns:
            TranslationDataset: An immutable dataset value object.
        """
        return cls(
            dataset_id=config.fingerprint,
            dataset_name=config.dataset_name,
            config=config,
            data=data,
            build_log=build_log,
        )
