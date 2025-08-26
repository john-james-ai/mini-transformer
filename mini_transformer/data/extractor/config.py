#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/extractor/config.py                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 09:53:53 am                                                 #
# Modified   : Monday August 25th 2025 09:55:44 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field

from mini_transformer.data.base import Config
from mini_transformer.utils.mixins import FingerprintMixin, ObjectRepresentationMixin


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class ExtractorConfig(Config, FingerprintMixin, ObjectRepresentationMixin):
    """Abstract base class for extractor configurations."""

    source: str = field(default="HuggingFace", metadata={"stable": True})
    source_dataset_name: str = field(default="wmt14", metadata={"stable": True})
    source_dataset_url: str = field(
        default="https://huggingface.co/datasets/wmt/wmt14", metadata={"stable": True}
    )
    split: str = field(default="train", metadata={"stable": True})
    n: int = field(default=400, metadata={"stable": True})
    stage: int = field(default="raw", metadata={"stable": True})
    shuffle: bool = field(default=True, metadata={"stable": True})
    buffer_size: int = field(default=30000, metadata={"stable": True})
    seed: int = field(default=42, metadata={"stable": True})


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class TranslationDatasetExtractorConfig(ExtractorConfig):

    lang: str = field(default="fr-en", metadata={"stable": True})
    lang_src: str = field(default="en", metadata={"stable": True})
    lang_tgt: str = field(default="fr", metadata={"stable": True})

    @property
    def name(self) -> str:
        return f"{self.source_dataset_name}-{self.lang}-{self.split}-{self.stage}-{self.n}_examples-{self.fingerprint}"
