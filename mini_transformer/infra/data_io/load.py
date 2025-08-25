#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/data/dataset/adapter.py                                           #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday August 24th 2025 11:43:12 pm                                                 #
# Modified   : Monday August 25th 2025 01:28:57 am                                                 #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
from typing import Any, Dict, List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as PyTorchDataset

# ------------------------------------------------------------------------------------------------ #


class PTDataset(PyTorchDataset):
    """A PyTorch-compatible Dataset that wraps a list of pre-tokenized data.

    This class acts as an adapter, taking a list of dictionaries (where each
    dictionary represents a single data point with fields like 'source_ids')
    and making it compatible with a `torch.utils.data.DataLoader`. It is
    responsible for retrieving a single item and converting its numerical lists
    into PyTorch tensors.

    Attributes:
        data (List[Dict[str, Any]]): A list where each element is a dictionary
            representing a pre-tokenized example.
    """

    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data

    def __len__(self) -> int:
        """Returns the total number of examples in the dataset.

        Returns:
            int: The total number of examples.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Retrieves an example by index and converts it to a tensor dictionary.

        Args:
            idx (int): The index of the example to retrieve.

        Returns:
            dict[str, torch.Tensor]: A dictionary where keys are field names
                (e.g., 'source_ids') and values are the corresponding token
                sequences as PyTorch tensors.
        """
        raw_item = self.data[idx]
        return {
            key: torch.tensor(value, dtype=torch.long)
            for key, value in raw_item.items()
        }


# ------------------------------------------------------------------------------------------------ #


def mt_collate_fn(batch):
    # 'batch' is a list of dictionaries, e.g., [ {'source_ids': tensor_1}, {'source_ids': tensor_2} ]
    source_ids = [item["source_ids"] for item in batch]
    target_input_ids = [item["target_input_ids"] for item in batch]
    target_label_ids = [item["target_label_ids"] for item in batch]

    # Use pad_sequence to handle padding for each key
    # padding_value=0 is a common choice for a padding token ID
    padded_sources = pad_sequence(source_ids, batch_first=True, padding_value=0)
    padded_target_inputs = pad_sequence(
        target_input_ids, batch_first=True, padding_value=0
    )
    padded_target_labels = pad_sequence(
        target_label_ids, batch_first=True, padding_value=0
    )

    # Create attention masks
    source_mask = padded_sources != 0

    return {
        "source_ids": padded_sources,
        "source_attention_mask": source_mask,
        "target_input_ids": padded_target_inputs,
        "target_label_ids": padded_target_labels,
    }
