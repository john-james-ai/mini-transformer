#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/base.py                                                      #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 06:52:31 am                                                 #
# Modified   : Monday August 25th 2025 09:21:23 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Abstract Base Classes for Dataset & DataFile Builders"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from mini_transformer.utils.mixins import FreezableMixin, ObjectRepresentationMixin


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DataObject(ABC, ObjectRepresentationMixin):
    # Metadata
    id: str  # Unique identifier for the dataset
    name: str  # Descriptive name for the dataset
    split: str  # train, validation, test
    n: int  # Number of rows in the dataset
    source: str  # Source ie HuggingFace
    source_dataset_name: str  # Dataset name at source

    # Dataset payload
    data: List[Dict[str, Any]]

    # Provenance
    created: datetime
    builder_config: BuilderConfig
    builder_metrics: BuilderMetrics

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
    def materialize(self, data: List[Dict[str, Any]]) -> DataObject:
        return replace(self, data=data)

    def dematerialize(self) -> DataObject:
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
class Builder(ABC):
    """Abstract base for Dataset and DataFile builders.

    Subclasses implement concrete extraction and construction logic for specific
    dataset types (e.g., translation pairs, classification records).

    Methods:
        build: Construct and return a concrete `Dataset` instance.
    """

    def __init__(self, builder_config: BuilderConfig) -> None:
        self._builder_config = builder_config

    @abstractmethod
    def build(self) -> Union[DataObject, DataObject]:
        """Build the data object.

        Returns:
            Union[Dataset, DataFile]: A concrete Dataset or DataFile instance

        Raises:
            ValueError: If the builder cannot construct a dataset due to
                insufficient valid examples or configuration issues.
        """


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class BuilderConfig(ABC):
    """Abstract base class for builder configurations."""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class BuilderMetrics(ABC, FreezableMixin, ObjectRepresentationMixin):
    """Abstract base class for builder metrics."""

    started_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    duration: float = 0.0

    def start(self) -> None:
        """Mark the beginning of metric collection.

        Sets ``started_at`` to the current wall-clock time.

        Notes:
            Calling :meth:`start` more than once will overwrite the previous
            ``started_at`` value.
        """
        self.started_at = datetime.now()

    def end(self) -> None:
        """Finalize metric collection and compute duration.

        Sets ``ended_at`` to the current wall-clock time and computes
        ``duration = round((ended_at - started_at).total_seconds(), 3)``.

        Raises:
            TypeError: If called before :meth:`start`.
        """
        if self.started_at is None:
            raise TypeError("Attempted to end without first calling start().")
        self.ended_at = datetime.now()
        self.duration = round((self.ended_at - self.started_at).total_seconds(), 3)

    def __enter__(self) -> "BuilderMetrics":
        """Enter the context manager and start timing.

        Returns:
            DatasetBuilderMetrics: The metrics object itself.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Exit the context manager, finalize timing, and log a summary.

        Always calls :meth:`end` and then :meth:`log_summary`. Exceptions
        raised within the context are **not** suppressed.
        """
        self.end()
        self.log_summary()

    @abstractmethod
    def log_summary(self) -> None:
        """Emit a concise, human-readable summary of the metrics.

        Implementations typically use the representation provided by
        ``ObjectRepresentationMixin`` (e.g., ``str(self)``) and log at
        INFO level. Called automatically from :meth:`__exit__`.
        """
        """Logs the builder_metrics."""
