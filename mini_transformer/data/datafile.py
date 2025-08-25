#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/datafile.py                                                  #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 11:17:40 pm                                                 #
# Modified   : Monday August 25th 2025 09:23:17 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List

from mini_transformer.data.base import DataObject
from mini_transformer.data.datafile_builder.config import (
    TranslationDataFileBuilderConfig,
)
from mini_transformer.data.datafile_builder.metrics import (
    TranslationDataFileBuilderMetrics,
)


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class TranslationDataFile(DataObject):
    lang: str  # Language code eg. "fr-en"
    lang_src: str  # Source language code eg. "en"
    lang_tgt: str  # Target language code eg. "fr"

    def name(self) -> str:
        return f"{self.source_dataset_name}-{self.lang}-{self.split}-{self.n}_examples-{self.id}"

    @classmethod
    def create(
        cls,
        builder_config: TranslationDataFileBuilderConfig,
        builder_metrics: TranslationDataFileBuilderMetrics,
        data: List[Dict[str, Any]],
    ) -> TranslationDataFile:
        """Creates a TranslationDataFile from a TranslationDataFileBuilderConfig.

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
