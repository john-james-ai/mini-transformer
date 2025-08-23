#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /mini_transformer/data/builder/base.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 11:57:08 pm                                                 #
# Modified   : Saturday August 23rd 2025 12:33:11 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime

from mini_transformer.data.dataset.base import Dataset
from mini_transformer.utils.mixins import FreezableMixin, ObjectRepresentationMixin


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DatasetBuilderConfig:
    """Abstract base class for dataset builder configurations."""

    source: str
    source_dataset_name: str
    source_dataset_url: str


# ------------------------------------------------------------------------------------------------ #
@dataclass
class DatasetBuilderMetrics(ABC, FreezableMixin, ObjectRepresentationMixin):
    """Lifecycle and timing metrics for dataset build runs.

    Tracks wall-clock timestamps and elapsed time for a single builder
    execution. Designed for use both directly (via :meth:`start` /
    :meth:`end`) and as a context manager. Subclasses should add
    domain-specific counters and implement :meth:`log_summary`.

    Mixins:
        FreezableMixin: Provides mechanisms to freeze the instance after
            finalization (e.g., via ``freeze()`` in subclasses).
        ObjectRepresentationMixin: Provides ``__str__``, ``__repr__``,
            ``as_dict()``, and ``export()`` helpers for logging and
            serialization.

    Attributes:
        started_at (datetime | None): Timestamp when :meth:`start` was
            called. ``None`` until started.
        ended_at (datetime | None): Timestamp when :meth:`end` completed.
            ``None`` until ended.
        duration (float): Elapsed seconds between ``started_at`` and
            ``ended_at``, rounded to three decimals. ``0.0`` until ended.
    """

    started_at: datetime = None
    ended_at: datetime = None
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

    def __enter__(self) -> "DatasetBuilderMetrics":
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


# ------------------------------------------------------------------------------------------------ #
class DatasetBuilder(ABC):
    """Abstract base for dataset builders.

    Subclasses implement concrete extraction and construction logic for specific
    dataset types (e.g., translation pairs, classification records).

    Methods:
        build: Construct and return a concrete `Dataset` instance.
    """

    @property
    @abstractmethod
    def builder_metrics(self) -> DatasetBuilderMetrics:
        """Returns an Metrics object"""

    @abstractmethod
    def build(self) -> Dataset:
        """Build the dataset.

        Returns:
            Dataset: A concrete dataset instance ready for downstream use
            (e.g., training, evaluation).

        Raises:
            ValueError: If the builder cannot construct a dataset due to
                insufficient valid examples or configuration issues.
        """
