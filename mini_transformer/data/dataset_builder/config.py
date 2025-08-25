#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/dataset_builder/config.py                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Friday August 22nd 2025 11:57:08 pm                                                 #
# Modified   : Monday August 25th 2025 08:41:30 am                                                 #
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
class TranslationDatasetBuilderConfig(
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
    n: int = field(default=120, metadata={"stable": True})
    oversample: int = field(default=10, metadata={"stable": True})
    tokens_min: int = field(default=8, metadata={"stable": True})
    tokens_max: int = field(default=256, metadata={"stable": True})
    ratio_min: float = field(default=0.5, metadata={"stable": True})
    ratio_max: float = field(default=2.0, metadata={"stable": True})
    src_bos: bool = field(default=False, metadata={"stable": True})
    src_eos: bool = field(default=False, metadata={"stable": True})
    tgt_bos: bool = field(default=True, metadata={"stable": True})
    tgt_eos: bool = field(default=True, metadata={"stable": True})
    seed: int = field(default=42, metadata={"stable": True})

    @property
    def name(self) -> str:
        return f"{self.source_dataset_name}-{self.lang}-{self.split}-{self.n}_examples-{self.fingerprint}"

    @property
    def src_max_words(self) -> int:
        """Compute the source-side word cap implied by ``tokens_max``.

        Returns:
            int: Word-count cap for source strings.
        """
        return self._compute_word_cap_from_tokens(bos=self.src_bos, eos=self.src_eos)

    @property
    def tgt_max_words(self) -> int:
        """Compute the target-side word cap implied by ``tokens_max``.

        Returns:
            int: Word-count cap for target strings.
        """
        return self._compute_word_cap_from_tokens(bos=self.tgt_bos, eos=self.tgt_eos)

    def _compute_word_cap_from_tokens(
        self, bos: bool, eos: bool, r: float = 1.5, margin: int = 8
    ) -> int:
        """Derive a word-count cap that rarely exceeds ``tokens_max`` after tokenization.

        This converts the model's token budget into a conservative word cap using
        a tokens-per-word estimate and a small margin for punctuation/edge cases.
        It lets you filter long strings *before* tokenization while keeping the
        vast majority of examples under the model's ``tokens_max``.

        Args:
            bos (bool): Whether a BOS token is added for this side.
            eos (bool): Whether an EOS token is added for this side.
            r (float): Estimated tokens per word (EN/FR with unigram SPM and
                smallish vocab ~ ``1.5``; use ``1.7`` for very small vocabs,
                ``1.3`` for larger).
            margin (int): Token headroom reserved for punctuation/oddities.

        Returns:
            int: Word-count cap to apply on raw text during extraction.
        """
        specials = int(bos) + int(eos)
        budget = self.tokens_max - specials - margin
        return max(1, budget // int(round(r)))
