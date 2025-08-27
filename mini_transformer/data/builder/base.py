#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/builder/base.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 26th 2025 10:02:18 pm                                                #
# Modified   : Tuesday August 26th 2025 11:55:40 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Abstract base classes for data builders."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from mini_transformer.data.dataset import Dataset
from mini_transformer.utils.mixins import (
    FingerprintMixin,
    FreezableMixin,
    ObjectRepresentationMixin,
)

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
class Builder(ABC):
    """Defines the abstract interface for all dataset builder classes.

    This class uses the Builder design pattern to provide a standardized,
    abstract interface for constructing complex dataset objects. Concrete
    implementations must provide the logic for the `build` method, which
    produces the final dataset, and expose a `metrics` property for tracking
    the build process.

    This pattern separates the complex construction logic of a dataset from its
    final representation, allowing for different build processes to create
    similar final objects.
    """

    @property
    @abstractmethod
    def metrics(self) -> MetricsCollector:
        """Metrics: Internal metrics with counters and timing."""
        pass

    @abstractmethod
    def build(self) -> Dataset:
        """Builds the object."""
        pass


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DatasetBuilderConfig(ABC, FingerprintMixin, ObjectRepresentationMixin):
    """Abstract base configuration for all dataset builders.

    This dataclass defines a common, abstract interface for dataset builder
    configurations. It holds standard parameters related to the data's source,
    the number of examples, and its processing stage. Concrete subclasses
    must implement the `dataset_name` property to provide a unique identifier
    for the final dataset.

    Attributes:
        source (str): The origin of the dataset (e.g., "HuggingFace").
        source_dataset_name (str): The canonical name of the dataset at its
            source (e.g., "wmt14").
        source_dataset_url (str): The URL where the source dataset can be found.
        split (str): The specific dataset split to use (e.g., "train").
        n (int): The target number of examples for the final dataset.
    """

    source: str = field(default="HuggingFace", metadata={"stable": True})
    source_dataset_name: str = field(default="wmt14", metadata={"stable": True})
    source_dataset_url: str = field(
        default="https://huggingface.co/datasets/wmt/wmt14", metadata={"stable": True}
    )
    split: str = field(default="train", metadata={"stable": True})
    n: int = field(default=50000, metadata={"stable": True})

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Name: Unique name for the dataset."""


# ------------------------------------------------------------------------------------------------ #
@dataclass
class MetricsCollector(ABC, FreezableMixin, ObjectRepresentationMixin):
    """Abstract base class for collecting and reporting process metrics.

    This class provides a standardized way to collect metrics, primarily for
    timing build processes. It can be used as a context manager to
    automatically track the duration of an operation, or by manually calling
    the `start()` and `end()` methods.

    As an abstract base class, it is intended to be subclassed to add more
    specific counters and metrics relevant to a particular process (e.g.,
    number of items processed, number of items filtered).

    Attributes:
        started_at (Optional[datetime]): The wall-clock time when metric
            collection began. Set by `start()`.
        ended_at (Optional[datetime]): The wall-clock time when metric
            collection ended. Set by `end()`.
        duration (float): The total duration of the process in seconds.
            Calculated by `end()`.
    """

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

    def __enter__(self) -> "MetricsCollector":
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

    def log_summary(self) -> None:
        """Log a compact, human-readable summary at INFO level.

        Emits ``str(self)`` via the module logger. Customize ``__str__`` in
        the base class if you need a different format.
        """
        output = str(self)
        logger.debug(output)
