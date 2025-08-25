#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/datafile_builder/config.py                                   #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 11:57:08 pm                                                 #
# Modified   : Monday August 25th 2025 08:41:45 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field

from mini_transformer.data.base import BuilderConfig
from mini_transformer.utils.mixins import FingerprintMixin, ObjectRepresentationMixin


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class TranslationDataFileBuilderConfig(
    BuilderConfig, FingerprintMixin, ObjectRepresentationMixin
):

    source: str = field(default="HuggingFace", metadata={"stable": True})
    source_dataset_name: str = field(default="wmt14", metadata={"stable": True})
    source_dataset_url: str = field(
        default="https://huggingface.co/datasets/wmt/wmt14", metadata={"stable": True}
    )
    lang: str = field(default="fr-en", metadata={"stable": True})
    lang_src: str = field(default="en", metadata={"stable": True})
    lang_tgt: str = field(default="fr", metadata={"stable": True})
    split: str = field(default="train", metadata={"stable": True})
    n: int = field(default=4000000, metadata={"stable": True})
    seed: int = field(default=42, metadata={"stable": True})

    @property
    def name(self) -> str:
        return f"{self.source_dataset_name}-{self.lang}-{self.split}-{self.n}_examples-{self.fingerprint}"
