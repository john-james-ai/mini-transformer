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
# Modified   : Monday August 25th 2025 09:18:42 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

from mini_transformer.data.base import DataObject
from mini_transformer.data.dataset_builder.metrics import (
    TranslationDatasetBuilderMetrics,
)

if TYPE_CHECKING:
    from mini_transformer.data.base import BuilderMetrics, BuilderConfig

from mini_transformer.data.dataset_builder.config import TranslationDatasetBuilderConfig


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class TranslationDataset(DataObject):
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
            "created": self.builder_metrics.ended_at,
        }

    @classmethod
    def create(
        cls,
        builder_config: TranslationDatasetBuilderConfig,
        builder_metrics: TranslationDatasetBuilderMetrics,
        data: List[Dict[str, Any]],
    ) -> TranslationDataset:
        """Creates a TranslationDataset from a TranslationDatasetBuilderConfig.

        Args:
            builder_config: The configuration to create the Dataset from.
            builder_metrics: The metrics collected during the build process.
            data: The dataset payload.

        Returns:
            TranslationDataset: The created Dataset.
        """
        return cls(
            id=builder_config.fingerprint,
            name=builder_config.name,
            split=builder_config.split,
            n=len(data),
            lang=builder_config.lang,
            lang_src=builder_config.lang_src,
            lang_tgt=builder_config.lang_tgt,
            source=builder_config.source,
            source_dataset_name=builder_config.source_dataset_name,
            data=data,
            created=datetime.now(),
            builder_config=builder_config,
            builder_metrics=builder_metrics,
        )
