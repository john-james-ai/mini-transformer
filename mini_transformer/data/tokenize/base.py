#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/tokenize/base.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 10:51:50 am                                                 #
# Modified   : Monday August 25th 2025 12:44:20 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from abc import ABC, abstractmethod


# ------------------------------------------------------------------------------------------------ #
class Tokenization(ABC):

    @property
    @abstractmethod
    def filepath(self) -> str:
        """Returns the file path to the tokenizer model."""

    @abstractmethod
    def tokenizer(self) -> str:
        """Returns the tokenizer model."""

    @abstractmethod
    def train(self, data: list[str]) -> None:
        """Trains the tokenizer on the provided data."""

    @abstractmethod
    def save(self) -> None:
        """Saves the tokenizer model to disk."""

    @abstractmethod
    def load(self) -> None:
        """Loads the tokenizer model from disk."""
