#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /config.py                                                                          #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Tuesday August 19th 2025 06:50:12 am                                                #
# Modified   : Tuesday August 19th 2025 06:15:53 pm                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass, field
from pathlib import Path

from mini_transformer.utils.mixins import FingerprintMixin


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DatasetConfig(FingerprintMixin):
    """Configuration for a small, reproducible dataset sample.

    This dataclass bundles dataset identity, split selection, sampling size,
    and simple string-level filters. Fields marked with ``metadata={"stable": True}``
    participate in the stable fingerprint computed by ``FingerprintMixin.fingerprint()``,
    which is then used to derive a deterministic manifest filename.

    Attributes:
        dataset (str): Hugging Face dataset name (e.g., "wmt14").
        lang (str): Language pair/config tag (e.g., "fr-en").
        split (str): Source split to draw from (e.g., "train", "validation", "test").
        k (int): Target number of examples to keep after filtering (often oversampled upstream).
        seed (int): RNG seed used for sampling/shuffling.
        min_words (int): Minimum word count per side (string-level filter).
        max_words (int): Maximum word count per side (string-level filter).
        ratio_min (float): Lower bound on length ratio max(len_en, len_fr)/min(...).
        ratio_max (float): Upper bound on length ratio max(len_en, len_fr)/min(...).
        lang_src (str): Source language key in records (e.g., "en").
        lang_tgt (str): Target language key in records (e.g., "fr").
        revision (str | None): Optional dataset/script revision tag recorded for provenance and
            included in the fingerprint when set.

    Properties:
        dataset_filename (str): Deterministic file name for the dataset.

    Example:
        >>> cfg = DatasetConfig(dataset="wmt14", lang="fr-en", split="train", k=160, seed=42)
        >>> cfg.dataset_filename
        'wmt14_fr-en_train_k160_seed42_ab12cd34.jsonl'
    """

    dataset: str = field(default="wmt14", metadata={"stable": True})
    lang: str = field(default="fr-en", metadata={"stable": True})
    split: str = field(default="train", metadata={"stable": True})
    k: int = field(default=160, metadata={"stable": True})
    seed: int = field(default=42, metadata={"stable": True})
    min_words: int = field(default=8, metadata={"stable": True})
    max_words: int = field(default=160, metadata={"stable": True})
    ratio_min: float = field(default=0.5, metadata={"stable": True})
    ratio_max: float = field(default=2.0, metadata={"stable": True})
    lang_src: str = field(default="en", metadata={"stable": True})
    lang_tgt: str = field(default="fr", metadata={"stable": True})
    revision: str | None = field(default=None, metadata={"stable": True})

    @property
    def dataset_filename(self) -> Path:
        """Build the deterministic dataset file name.

        Returns:
            Path like
            ``wmt14_fr-en_train_k160_seed42_<fingerprint>.jsonl``.

        Notes:
            The fingerprint is computed from fields marked as ``stable`` by
            :meth:`FingerprintMixin.fingerprint`.
        """
        return Path(
            f"{self.dataset}_{self.lang}_{self.split}_k{self.k}_seed{self.seed}_{self.fingerprint}.jsonl"
        )


# ------------------------------------------------------------------------------------------------ #
