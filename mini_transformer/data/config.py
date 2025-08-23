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
# Modified   : Friday August 22nd 2025 07:32:27 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from dataclasses import dataclass, field

from mini_transformer.utils.mixins import FingerprintMixin, ObjectRepresentationMixin


# ------------------------------------------------------------------------------------------------ #
@dataclass(frozen=True)
class DatasetConfig(FingerprintMixin, ObjectRepresentationMixin):
    """Configuration for building a small, reproducible WMT-style sample.

    This dataclass specifies dataset identity, split selection, target sample size,
    and simple length/ratio filters. Fields marked with ``metadata={"stable": True}``
    participate in the stable fingerprint computed by
    :meth:`FingerprintMixin.fingerprint`, which is then used to derive a
    deterministic output name.

    Attributes:
        dataset (str): Hugging Face dataset name (e.g., ``"wmt14"``).
        lang (str): Language-pair/config tag (e.g., ``"fr-en"``).
        lang_src (str): Source language key in records (e.g., ``"en"``).
        lang_tgt (str): Target language key in records (e.g., ``"fr"``).
        split (str): Source split to draw from (e.g., ``"train"``, ``"validation"``, ``"test"``).
        dataset_size (int): Number of final examples to keep **after** filtering.
        oversample (int): Oversampling factor used to collect a larger candidate pool
            before tokenization and final caps. For example, ``oversample=10`` means
            sample approximately ``10 × dataset_size`` candidates, then drop those
            that exceed token caps and keep the first ``dataset_size`` that fit.
        tokens_min (int): Minimum **token** count per side allowed at training time.
            Typically a loose lower bound to exclude extremely short sequences.
        tokens_max (int): Maximum **token** count per side allowed at training time
            (the model's sequence budget, e.g., ``256``). Also used to derive
            word-level caps during extraction via
            :meth:`_compute_word_cap_from_tokens`.
        ratio_min (float): Lower bound on length ratio
            ``max(len_src, len_tgt) / min(len_src, len_tgt)`` (word counts).
        ratio_max (float): Upper bound on the same length ratio (word counts).
        src_bos (bool): Whether **source** sequences include a BOS token.
        src_eos (bool): Whether **source** sequences include an EOS token.
        tgt_bos (bool): Whether **target** sequences include a BOS token
            (commonly ``True`` for seq2seq training).
        tgt_eos (bool): Whether **target** sequences include an EOS token
            (commonly ``True`` for seq2seq training).
        revision (str | None): Optional dataset/script revision recorded for
            provenance; included in the fingerprint when set.
        seed (int): RNG seed used for sampling/shuffling.

    Properties:
        dataset_name (Path): Deterministic name derived from stable
            fields and the fingerprint (JSONL, one record per line).
        src_max_words (int): Word-count cap for source strings computed from
            ``tokens_max`` and BOS/EOS flags (fudge-factor based).
        tgt_max_words (int): Word-count cap for target strings computed from
            ``tokens_max`` and BOS/EOS flags.

    Example:
        >>> cfg = DatasetConfig(dataset="wmt14", lang="fr-en", split="train",
        ...                     dataset_size=160, seed=42)
        >>> cfg.dataset_name.name  # doctest: +SKIP
        'wmt14_fr-en_train_k160_seed42_ab12cd34.jsonl'
    """

    dataset: str = field(default="wmt14", metadata={"stable": True})
    lang: str = field(default="fr-en", metadata={"stable": True})
    lang_src: str = field(default="en", metadata={"stable": True})
    lang_tgt: str = field(default="fr", metadata={"stable": True})
    split: str = field(default="train", metadata={"stable": True})
    dataset_size: int = field(default=120, metadata={"stable": True})
    oversample: int = field(default=10, metadata={"stable": True})
    tokens_min: int = field(default=8, metadata={"stable": True})
    tokens_max: int = field(default=256, metadata={"stable": True})
    ratio_min: float = field(default=0.5, metadata={"stable": True})
    ratio_max: float = field(default=2.0, metadata={"stable": True})
    src_bos: bool = field(default=False, metadata={"stable": True})
    src_eos: bool = field(default=False, metadata={"stable": True})
    tgt_bos: bool = field(default=True, metadata={"stable": True})
    tgt_eos: bool = field(default=True, metadata={"stable": True})
    revision: str | None = field(default=None, metadata={"stable": True})
    seed: int = field(default=42, metadata={"stable": True})

    @property
    def dataset_name(self) -> str:
        """Build the deterministic dataset name.

        Returns: str of the form
            ``{dataset}_{lang}_{split}_k{dataset_size}_seed{seed}_{fingerprint}``.

        Notes:
            The fingerprint is computed from fields marked as ``stable`` by
            :meth:`FingerprintMixin.fingerprint`.
        """
        return f"{self.dataset}_{self.lang}_{self.split}_k{self.dataset_size}_seed{self.seed}_{self.fingerprint}"

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
