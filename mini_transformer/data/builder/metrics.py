#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/builder/metrics.py                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 11:57:08 pm                                                 #
# Modified   : Monday August 25th 2025 02:41:23 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
from abc import ABC
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from mini_transformer.utils.mixins import FreezableMixin, ObjectRepresentationMixin

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


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

    def log_summary(self) -> None:
        """Log a compact, human-readable summary at INFO level.

        Emits ``str(self)`` via the module logger. Customize ``__str__`` in
        the base class if you need a different format.
        """
        output = str(self)
        logger.debug(output)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TranslationDatasetBuilderMetrics(BuilderMetrics):
    n: int = 0

    seen: int = 0
    filtered: int = 0
    candidates: int = 0
    selected: int = 0

    filtered_empty: int = 0
    filtered_ratio: int = 0
    filtered_src_short: int = 0
    filtered_src_long: int = 0
    filtered_tgt_short: int = 0
    filtered_tgt_long: int = 0

    throughput: float = 0.0

    def end(self) -> None:
        """Finalize metrics, compute throughput, and freeze the snapshot.

        Calls :meth:`super().end()` to close timing and set ``duration``,
        computes ``throughput = round(seen / duration, 3)`` when duration
        is positive (else 0.0), and invokes :meth:`freeze` to prevent
        further mutation.
        """
        super().end()
        d = self.duration
        self.throughput = round(self.seen / d, 3) if d > 0 else 0.0
        self.freeze()
