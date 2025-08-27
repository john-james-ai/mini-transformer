#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /tests/test_data/test_tokenization.py                                               #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 04:44:11 pm                                                 #
# Modified   : Wednesday August 27th 2025 12:11:50 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import inspect
import logging
import os
from datetime import datetime

import pytest
import tokenizers

# ------------------------------------------------------------------------------------------------ #
# pylint: disable=missing-class-docstring, line-too-long
# mypy: ignore-errors
# ------------------------------------------------------------------------------------------------ #
# ------------------------------------------------------------------------------------------------ #
logger = logging.getLogger(__name__)
# ------------------------------------------------------------------------------------------------ #
double_line = f"\n{100 * '='}"
single_line = f"\n{100 * '-'}"


@pytest.mark.token
class TestTokenization:  # pragma: no cover
    # ============================================================================================ #
    def test_tokenization_training(self, container, dataset, caplog) -> None:
        start = datetime.now()
        logger.info(
            f"\n\nStarted {self.__class__.__name__} {inspect.stack()[0][3]} at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(double_line)
        # ---------------------------------------------------------------------------------------- #
        tokenization = container.data.tokenization()
        tokenization.train(dataset=dataset)
        tok1 = tokenization.tokenizer
        assert isinstance(tokenization.tokenizer, tokenizers.Tokenizer)
        assert isinstance(tokenization.filepath, str)
        assert isinstance(tokenization._vocab_size, int)
        assert tokenization._vocab_size == 30000  # From config.yaml
        tokenization.save()
        loaded_tokenization = container.data.tokenization()
        assert isinstance(loaded_tokenization.filepath, str)
        assert isinstance(loaded_tokenization._vocab_size, int)
        assert loaded_tokenization._vocab_size == 30000  # From config.yaml
        assert os.path.exists(loaded_tokenization.filepath)
        loaded_tokenization.load()
        tok2 = tokenization.tokenizer

        # 1) behavioral equivalence
        samples = ["hello world", "bonjour le monde", "<BOS> hello <EOS>"]
        for s in samples:
            assert tok1.encode(s).ids == tok2.encode(s).ids

        # 2) Check vocab equivalence
        v1 = tok1.get_vocab()  # token -> id
        v2 = tok2.get_vocab()
        assert v1 == v2, "vocab differs"

        # 3) specials mapped the same
        for t in ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]:
            if t in v1:
                assert tok1.token_to_id(t) == tok2.token_to_id(t)

        # 4) component sanity
        assert tok1.model.__class__ is tok2.model.__class__
        assert type(tok1.pre_tokenizer).__name__ == type(tok2.pre_tokenizer).__name__
        # ---------------------------------------------------------------------------------------- #
        end = datetime.now()
        duration = round((end - start).total_seconds(), 1)

        logger.info(
            f"\n\nCompleted {self.__class__.__name__} {inspect.stack()[0][3]} in {duration} seconds at {start.strftime('%I:%M:%S %p')} on {start.strftime('%m/%d/%Y')}"
        )
        logger.info(single_line)
