#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/datafile_builder/metrics.py                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 07:50:19 am                                                 #
# Modified   : Monday August 25th 2025 08:27:20 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from mini_transformer.data.base import BuilderMetrics
from mini_transformer.utils.mixins import FreezableMixin, ObjectRepresentationMixin

# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------------------------------------ #
@dataclass
class TranslationDataFileBuilderMetrics(
    BuilderMetrics, FreezableMixin, ObjectRepresentationMixin
):

    n: int = 0

    avg_seq_len: float = 0.0
    max_seq_len: int = 0
    min_seq_len: Optional[int] = 0

    src_avg_seq_len: float = 0.0
    src_max_seq_len: int = 0
    src_min_seq_len: int = 0

    tgt_avg_seq_len: float = 0.0
    tgt_max_seq_len: int = 0
    tgt_min_seq_len: int = 0

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
        self.throughput = round(self.n / d, 3) if d > 0 else 0.0
        self.freeze()

    def log_summary(self) -> None:
        """Log a compact, human-readable summary at INFO level.

        Emits ``str(self)`` via the module logger. Customize ``__str__`` in
        the base class if you need a different format.
        """
        output = str(self)
        logger.info(output)
