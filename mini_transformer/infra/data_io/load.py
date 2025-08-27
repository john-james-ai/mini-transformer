#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.11.13                                                                             #
# Filename   : /mini_transformer/infra/data_io/load.py                                             #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Sunday August 24th 2025 11:43:12 pm                                                 #
# Modified   : Tuesday August 26th 2025 12:12:29 am                                                #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import collections
import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset as PyTorchDataset
from torch.utils.data import Sampler

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


def collate_fn(batch, tok, L_src=64, L_tgt=64, as_numpy=True):
    # batch: list of {"src": str, "tgt": str}
    PAD = tok.token_to_id("<PAD>")
    BOS = tok.token_to_id("<BOS>")
    EOS = tok.token_to_id("<EOS>")
    assert None not in (PAD, BOS, EOS)

    src_texts = [b["src"] for b in batch]
    tgt_texts = [b["tgt"] for b in batch]

    # Encoder side
    tok.enable_truncation(max_length=L_src)
    tok.enable_padding(length=L_src, pad_id=PAD, pad_token="<PAD>")
    enc_src = tok.encode_batch(src_texts)  # fixed length L_src
    src_ids = [e.ids for e in enc_src]  # (B, L_src)
    Lb_src = max(len(s) for s in src_ids)

    # Decoder side
    tok.enable_truncation(max_length=L_tgt)
    tok.enable_padding(length=L_tgt, pad_id=PAD, pad_token="<PAD>")
    enc_tgt = tok.encode_batch(tgt_texts)  # fixed length L_tgt
    tgt_ids = [e.ids for e in enc_tgt]  # (B, L_tgt)
    Lb_tgt = max(len(t) for t in tgt_ids)

    # Build decoder inputs / labels (shift-right)
    dec_in = [[BOS] + t[: Lb_tgt - 1] for t in tgt_ids]  # (B, L_tgt)
    labels = [t[: Lb_tgt - 1] + [EOS] for t in tgt_ids]  # (B, L_tgt)

    # pad to the batch max
    pad = lambda seqs, L: [s + [PAD] * (L - len(s)) for s in seqs]
    src_ids = pad(src_ids, Lb_src)
    dec_in = pad(dec_in, Lb_tgt)
    labels = pad(labels, Lb_tgt)

    # Masks (1 = real token, 0 = PAD)
    enc_mask = [[1 if x != PAD else 0 for x in s] for s in src_ids]
    dec_mask = [[1 if x != PAD else 0 for x in d] for d in dec_in]

    if as_numpy:
        import numpy as np

        src_ids = np.asarray(src_ids, dtype=np.int32)
        dec_in = np.asarray(dec_in, dtype=np.int32)
        labels = np.asarray(labels, dtype=np.int32)
        enc_mask = np.asarray(enc_mask, dtype=bool)
        dec_mask = np.asarray(dec_mask, dtype=bool)
    return {
        "encoder_input_ids": src_ids,
        "decoder_input_ids": dec_in,
        "labels": labels,
        "encoder_attn_mask": enc_mask,
        "decoder_attn_mask": dec_mask,
        "pad_id": PAD,
    }


# ------------------------------------------------------------------------------------------------ #


class BucketBatchSampler(Sampler):
    def __init__(
        self,
        bucket_ids: list,
        batch_size: int = 16,
        len_key: list = [],
        drop_last: bool = False,
        seed: int = 42,
    ):
        self.bucket_ids = bucket_ids
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

        # 1) make bucket ids once (or on init)
        bucket_ids = [min((k - 1) // 16, 3) for k in len_key]  # for L=64, 4 buckets
        self.bucket_to_idx = collections.defaultdict(list)
        for i, b in enumerate(bucket_ids):
            self.bucket_to_idx[b].append(i)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        buckets = {
            b: rng.sample(ixs, len(ixs)) for b, ixs in self.bucket_to_idx.items()
        }
        order = []
        while any(buckets.values()):
            for b in sorted(buckets):
                buf = buckets[b]
                if not buf:
                    continue
                batch, buckets[b] = buf[: self.batch_size], buf[self.batch_size :]
                if batch and (len(batch) == self.batch_size or not self.drop_last):
                    yield batch
        self.epoch += 1

    def __len__(self):
        n = sum(len(v) for v in self.bucket_to_idx.values())
        return n // self.batch_size  # approx, ignores last partials
