#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/infra/data_io/download.py                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 25th 2025 02:42:28 am                                                 #
# Modified   : Monday August 25th 2025 03:53:00 pm                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
"""Hugging Face Dataset Downloader"""

from typing import Optional

from datasets import IterableDataset, load_dataset

# ------------------------------------------------------------------------------------------------ #


class HFDatasetDownloader:
    """Streams datasets from the Hugging Face Hub.

    This class provides a configurable interface to stream dataset splits
    from Hugging Face, with options for shuffling and taking a subset of
    the data. It is designed to work with large datasets by not downloading
    the entire dataset at once.

    Attributes:
        dataset (str): The name of the dataset on the Hugging Face Hub.
        language (str): The language configuration or subset of the dataset.
        shuffle (bool): If True, shuffles the dataset using a buffer.
        buffer_size (int): The size of the buffer to use for shuffling.
        seed (Optional[int]): A random seed for reproducible shuffling.
    """

    def __init__(
        self,
        dataset: str,
        language: str,
        split: str,
        n: int,
        shuffle: bool = True,
        buffer_size: int = 10000,
        seed: Optional[int] = None,
    ) -> None:
        self._dataset = dataset
        self._language = language
        self._split = split
        self._n = n
        self._shuffle = shuffle
        self._buffer_size = buffer_size
        self._seed = seed

    def download(self) -> IterableDataset:
        """Downloads the dataset from Hugging Face and returns an iterable dataset."""

        dataset: IterableDataset = load_dataset(
            self._dataset,
            self._language,
            split=self._split,
            streaming=True,
        )

        if self._shuffle:
            dataset = dataset.shuffle(buffer_size=self._buffer_size, seed=self._seed)

        if self._n is not None:
            dataset = dataset.take(self._n)

        return dataset
