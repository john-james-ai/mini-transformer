#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/tokenize/bpe.py                                              #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 10:59:59 am                                                 #
# Modified   : Monday August 25th 2025 06:30:31 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
# %%
from pathlib import Path
from typing import List, Optional

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

from mini_transformer.data.dataset import Dataset, TranslationDataset
from mini_transformer.data.tokenize.base import Tokenization

# ------------------------------------------------------------------------------------------------ #


class BPETokenization(Tokenization):
    """Lightweight wrapper around Hugging Face `tokenizers` for BPE training.

    This class encapsulates training, saving, and loading of a Byte Pair
    Encoding (BPE) tokenizer using the `tokenizers` library. It is intended
    for simple, reproducible experiments where you control vocabulary size,
    frequency thresholds, and special tokens, and where training text is
    derived from a parallel translation dataset.

    Attributes:
        _filepath: Filesystem path where the tokenizer JSON is saved/loaded.
        _vocab_size: Target vocabulary size for the BPE trainer.
        _min_frequency: Minimum token frequency required to enter the vocab.
        _show_progress: Whether to display a progress bar during training.
        _unk_token: String used for unknown pieces (must exist in special tokens).
        _special_tokens: List of reserved tokens injected into the vocabulary.
        _tokenizer: The underlying `tokenizers.Tokenizer` instance.
        _data: Internal buffer for prepared text (currently unused).
    """

    def __init__(
        self,
        filepath: str,
        vocab_size: int = 50000,
        min_frequency: int = 2,
        show_progress: bool = True,
        unk_token: str = "<UNK>",
        special_tokens: Optional[list] = None,
    ) -> None:
        """Initialize a BPE tokenizer configuration.

        Args:
            filepath: Destination path for the serialized tokenizer JSON file.
            vocab_size: Target vocabulary size for the BPE trainer.
            min_frequency: Minimum frequency threshold for inclusion in the vocab.
            show_progress: If True, display a progress bar during training.
            unk_token: Token to represent out-of-vocabulary pieces.
            special_tokens: Explicit list of special tokens to reserve. If None,
                defaults to ``["<PAD>", "<BOS>", "<EOS>", "<UNK>"]``.

        Notes:
            This constructor creates the underlying `Tokenizer` instance with a
            `BPE` model configured to use ``unk_token`` for unknown tokens.

        """
        self._filepath = filepath
        self._vocab_size = vocab_size
        self._min_frequency = min_frequency
        self._show_progress = show_progress
        self._unk_token = unk_token
        self._special_tokens = (
            special_tokens
            if special_tokens is not None
            else ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        )
        self._tokenizer = Tokenizer(BPE(unk_token=self._unk_token))
        self._data = []

    @property
    def filepath(self) -> str:
        """Path to the tokenizer JSON artifact.

        Returns:
            The file path used by :meth:`save` and :meth:`load`.
        """
        return self._filepath

    @property
    def tokenizer(self) -> Optional[Tokenizer]:
        """Return the underlying `tokenizers.Tokenizer` instance.

        Returns:
            The trained/loaded tokenizer, or ``None`` if not yet initialized.

        Notes:
            Returning the live tokenizer allows advanced operations (e.g.,
            ``encode``, ``decode``, ``token_to_id``). Treat as read-mostly if
            accessed from multiple threads.
        """
        return self._tokenizer

    def train(self, dataset: TranslationDataset) -> None:
        """Train the BPE tokenizer on text extracted from a translation dataset.

        Args:
            dataset: A translation-style dataset providing:
                - ``data``: an iterable of dict-like rows. Each row is either:
                  * a mapping with a ``"translation"`` key whose value is a
                    nested mapping of language codes to strings, or
                  * a flat mapping where the source/target texts are directly
                    under the language keys.
                - ``config.lang_src`` and ``config.lang_tgt``: language codes
                  (e.g., ``"en"``, ``"fr"``) used to select source/target text.

        Raises:
            ValueError: If the prepared text list is empty.
            RuntimeError: If training fails within the underlying library.

        Example:
            >>> tok = BPETokenization("artifacts/bpe.json", vocab_size=8000)
            >>> tok.train(dataset)  # dataset provides data + config.lang_src/tgt
        """
        data = self._prepare_data(dataset=dataset)
        if not data:
            raise ValueError("No training text found in dataset.")

        print(f"Second validation of parameters before training...")
        self._validate()

        # Instantiate a trainer
        trainer = BpeTrainer(
            vocab_size=self._vocab_size,  # type: ignore[reportCallIssue]
            min_frequency=self._min_frequency,  # type: ignore[reportCallIssue]
            special_tokens=self._special_tokens,  # type: ignore[reportCallIssue]
            show_progress=self._show_progress,  # type: ignore[reportCallIssue]
        )

        # Use whitespace to tokenize the input
        self._tokenizer.pre_tokenizer = Whitespace()  # type: ignore[assignment]

        try:
            self._tokenizer.train_from_iterator(data, trainer=trainer)
        except Exception as e:
            raise RuntimeError("BPE training failed.") from e

    def save(self) -> None:
        """Serialize the tokenizer to :pyattr:`filepath` in JSON format.

        Saves the current tokenizer state to disk. Existing files at the same
        path may be overwritten.

        Raises:
            OSError: If the file cannot be written (e.g., permissions, missing directory).
        """
        filepath = Path(self._filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)  # type: ignore[attr-defined]
        self._tokenizer.save(self._filepath)

    def load(self) -> None:
        """Load a tokenizer from :pyattr:`filepath` into this instance.

        After calling this method, :meth:`tokenizer` returns the loaded model.

        Raises:
            FileNotFoundError: If :pyattr:`filepath` does not exist.
            OSError: If the file cannot be read or parsed.
        """
        self._tokenizer = Tokenizer.from_file(self._filepath)

    def _prepare_data(self, dataset: Dataset) -> List[str]:
        """Extract source and target texts from the dataset for tokenizer training.

        This helper flattens a translation-style dataset into a simple list of
        strings by pulling both the source and target language texts for each row.

        Args:
            dataset: A dataset object with:
                - ``data``: iterable of row dicts (see formats above).
                - ``config.lang_src`` and ``config.lang_tgt``: language keys.

        Returns:
            A list of strings containing interleaved source and target texts,
            suitable for consumption by `Tokenizer.train_from_iterator`.

        Notes:
            Rows lacking either language key will contribute an empty string
            for that side. You may wish to pre-filter such rows upstream.
        """
        texts = []
        # Prepare the data for training the tokenizer
        for row in dataset.data:
            row = row.get("translation", row)
            texts.append(row.get(dataset.config.lang_src, ""))
            texts.append(row.get(dataset.config.lang_tgt, ""))
        return texts

    def _validate(self) -> None:
        # vocab_size
        if self._vocab_size is None:
            raise ValueError("vocab_size cannot be None.")
        try:
            self._vocab_size = int(self._vocab_size)
        except Exception:
            raise TypeError(
                f"vocab_size must be int, got {type(self._vocab_size).__name__}"
            )
        if self._vocab_size < 1:
            raise ValueError("vocab_size must be >= 1.")

        # min_frequency
        if self._min_frequency is None:
            raise ValueError("min_frequency cannot be None.")
        try:
            self._min_frequency = int(self._min_frequency)
        except Exception:
            raise TypeError(
                f"min_frequency must be int, got {type(self._min_frequency).__name__}"
            )
        if self._min_frequency < 1:
            raise ValueError("min_frequency must be >= 1.")

        # specials
        if not self._special_tokens:
            raise ValueError("special_tokens cannot be empty.")
        if self._unk_token not in self._special_tokens:
            raise ValueError("unk_token must be present in special_tokens.")
        if not all(isinstance(t, str) for t in self._special_tokens):
            raise TypeError("All special_tokens must be strings.")
