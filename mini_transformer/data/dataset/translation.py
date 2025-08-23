#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /mini_transformer/data/dataset/translation.py                                       #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 11:34:55 pm                                                 #
# Modified   : Saturday August 23rd 2025 12:44:57 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

from mini_transformer.data.dataset.base import Dataset

if TYPE_CHECKING:
    from mini_transformer.data.builder.translation import (
        TranslationDatasetBuilderConfig,
        TranslationDatasetBuilderMetrics,
    )


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
            "size": self.size,
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
        """Construct a `TranslationDataset` from builder outputs.

        This helper assigns `dataset_id` from the configuration fingerprint,
        attaches the frozen `builder_metrics`, and stores the provided `data`
        unchanged.

        Args:
            config: The frozen `TranslationDatasetBuilderConfig` used to produce `data`. Must
                expose a `fingerprint` accessor (property or method) that
                yields a stable identifier.
            builder_metrics: The finalized, frozen builder_metrics produced by the builder,
                containing counts and timing for this build.
            data: Materialized translation pairs as a list of dictionaries
                (each with keys like {"idx", "src", "tgt"}).

        Returns:
            TranslationDataset: An immutable dataset value object.
        """
        size = len(data)
        created = datetime.now()
        name = f"{builder_config.source_dataset_name}_{builder_config.lang}_{builder_config.split}_{size}-rows_seed-{builder_config.seed}_{builder_config.fingerprint}"
        return cls(
            id=builder_config.fingerprint,
            name=name,
            split=builder_config.split,
            lang=builder_config.lang,
            lang_src=builder_config.lang_src,
            lang_tgt=builder_config.lang_tgt,
            size=size,
            source=builder_config.source,
            source_dataset_name=builder_config.source_dataset_name,
            data=data,
            builder_config=builder_config,
            builder_metrics=builder_metrics,
            created=created,
        )
