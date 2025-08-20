#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /extract.py                                                                         #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Monday August 18th 2025 11:59:17 pm                                                 #
# Modified   : Wednesday August 20th 2025 12:24:05 am                                              #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from __future__ import annotations

import os
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, List

from datasets import load_dataset
from datasets.iterable_dataset import IterableDataset
from dotenv import load_dotenv

from mini_transformer.data.config import DatasetConfig

# ------------------------------------------------------------------------------------------------ #
load_dotenv()
# ------------------------------------------------------------------------------------------------ #
DATA_ROOT = Path(os.getenv("DATA_ROOT", "data")).expanduser()


# ------------------------------------------------------------------------------------------------ #
class DatasetExtractor:
    def __init__(self, config: DatasetConfig) -> None:
        self._config = config

    def extract(self, force: bool = False) -> None:
        if not self.exists or force:
            stream = load_dataset(
                self._config.dataset,
                self._config.subset,
                split=self._config.split,
                streaming=self._config.streaming,
            )
            return self._sample(stream)  # type: ignore[reportArgumentType]

    def exists(self) -> bool:

        return os.path.exists(self._config.dataset_dir)

    def _sample(
        self,
        iterable: IterableDataset,
    ) -> List[Dict[str, Any]]:
        i = 0
        data = []
        for row in iterable:
            if i < self._config.k:
                parsed_row = self._parse_row(row)
                if self._is_valid(parsed_row):  # type: ignore[reportArgumentType]
                    data.append(parsed_row)
                    i += 0
            else:
                break
        return data

    def _is_valid(self, row: Dict[str, Any]) -> bool:
        len_src = len(row["src"].split())
        len_tgt = len(row["tgt"].split())

        return (
            self._config.min_words <= len_src <= self._config.max_words
            and self._config.min_words <= len_tgt <= self._config.max_words
            and self._config.min_src_tgt_ratio
            <= len_src / len_tgt
            <= self._config.max_src_tgt_ratio
        )

    def _parse_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        data = {
            "src": row["translation"][self._config.lang_src],
            "tgt": row["translation"][self._config.lang_tgt],
        }
        row_id = sha1((repr(data)).encode("utf-8")).hexdigest()[:6]
        return {"row_id": row_id, "src": data["src"], "tgt": data["tgt"]}
