#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# ================================================================================================ #
# Project    : Mini-Transformer                                                                    #
# Version    : 0.1.0                                                                               #
# Python     : 3.13.5                                                                              #
# Filename   : /tests/test_data/test_builder.py                                                    #
# ------------------------------------------------------------------------------------------------ #
# Author     : John James                                                                          #
# Email      : john.james.ai.studio@gmail.com                                                      #
# URL        : https://github.com/john-james-ai/mini-transformer                                   #
# ------------------------------------------------------------------------------------------------ #
# Created    : Thursday August 21st 2025 06:10:44 pm                                               #
# Modified   : Saturday August 23rd 2025 12:54:32 am                                               #
# ------------------------------------------------------------------------------------------------ #
# License    : MIT License                                                                         #
# Copyright  : (c) 2025 John James                                                                 #
# ================================================================================================ #
import copy

import pytest

# ⬇️ adjust these imports to your project layout
from mini_transformer.data.builder.translation import (
    TranslationDatasetBuilder,
    TranslationDatasetBuilderConfig,
)

# ---------- helpers ----------


def _words(n: int) -> str:
    return "w " * (n - 1) + "w" if n > 0 else ""


def stream_all_valid(n: int, lang_src="en", lang_tgt="fr", nested=True):
    """Yield n valid rows."""
    for i in range(n):
        src = f"w w w w w{i}"
        tgt = f"w w w w w{i}"
        payload = {lang_src: src, lang_tgt: tgt}
        yield {"translation": payload} if nested else payload


def stream_missing_key_first(lang_src="en"):
    """First row missing target key to trigger KeyError."""
    yield {"translation": {lang_src: _words(3)}}  # missing other lang


def stream_known_mix_for_counts():
    """5 invalid upfront (3 empty, 2 ratio) then 10 valid."""
    # 3 empty
    for _ in range(3):
        yield {"translation": {"en": "", "fr": _words(3)}}
    # 2 bad ratio (e.g., 2 vs 10 with ratio_max=2.0)
    for _ in range(2):
        yield {"translation": {"en": _words(2), "fr": _words(10)}}
    # 10 valid
    for _ in range(10):
        yield {"translation": {"en": _words(4), "fr": _words(4)}}


def stream_too_few_valid():
    """Only 6 valid total; will fail if dataset_target_size=8."""
    for _ in range(6):
        yield {"translation": {"en": _words(4), "fr": _words(4)}}
    # lots of invalid after
    for _ in range(50):
        yield {"translation": {"en": "", "fr": _words(3)}}


def make_config(
    dataset_target_size=8,
    oversample=5,
    seed=123,
    tokens_min=1,
    tokens_max=10,
    ratio_max=2.0,
    lang_src="en",
    lang_tgt="fr",
):
    return TranslationDatasetBuilderConfig(
        source_dataset_name="wmt14",  # not used (we monkeypatch _load_dataset)
        lang=f"{lang_src}-{lang_tgt}",
        split="train",
        dataset_target_size=dataset_target_size,
        oversample=oversample,
        seed=seed,
        lang_src=lang_src,
        lang_tgt=lang_tgt,
        tokens_min=tokens_min,
        tokens_max=tokens_max,
        ratio_max=ratio_max,
    )


# ---------- tests ----------


@pytest.mark.builder
def test_early_stop_at_target_valid(monkeypatch):
    """Stops once candidates reach target; builder_metrics counters reflect that."""
    cfg = make_config(dataset_target_size=8, oversample=5)  # target = 40
    b = TranslationDatasetBuilder(cfg)
    monkeypatch.setattr(b, "_load_dataset", lambda: stream_all_valid(100))

    out = b._extract(b._load_dataset())

    assert len(out) == cfg.dataset_target_size
    # all rows valid → seen == candidates == target
    assert b.builder_metrics.seen == cfg.dataset_target_size * cfg.oversample
    assert b.builder_metrics.candidates == cfg.dataset_target_size * cfg.oversample
    assert b.builder_metrics.filtered == 0
    assert b.builder_metrics.selected == cfg.dataset_target_size


@pytest.mark.builder
def test_deterministic_sampling_same_seed(monkeypatch):
    """Same seed + same stream order ⇒ identical selection and order."""
    cfg = make_config(seed=777)
    b1 = TranslationDatasetBuilder(cfg)
    b2 = TranslationDatasetBuilder(copy.deepcopy(cfg))

    monkeypatch.setattr(b1, "_load_dataset", lambda: stream_all_valid(100))
    monkeypatch.setattr(b2, "_load_dataset", lambda: stream_all_valid(100))

    out1 = b1._extract(b1._load_dataset())
    out2 = b2._extract(b2._load_dataset())

    assert out1 == out2


@pytest.mark.builder
def test_seed_changes_selection(monkeypatch):
    """Different seeds ⇒ different selections (almost surely)."""
    cfg_a = make_config(seed=1)
    cfg_b = make_config(seed=2)

    a = TranslationDatasetBuilder(cfg_a)
    b = TranslationDatasetBuilder(cfg_b)

    monkeypatch.setattr(a, "_load_dataset", lambda: stream_all_valid(100))
    monkeypatch.setattr(b, "_load_dataset", lambda: stream_all_valid(100))

    out_a = a._extract(a._load_dataset())
    out_b = b._extract(b._load_dataset())

    assert {r["idx"] for r in out_a} != {r["idx"] for r in out_b}


@pytest.mark.builder
def test_keyerror_on_missing_language_key(monkeypatch):
    """Missing lang keys should raise KeyError with context."""
    cfg = make_config()
    b = TranslationDatasetBuilder(cfg)
    monkeypatch.setattr(b, "_load_dataset", lambda: stream_missing_key_first())

    with pytest.raises(KeyError) as exc:
        _ = b._extract(b._load_dataset())
    msg = str(exc.value)
    assert cfg.lang_src in msg and cfg.lang_tgt in msg


@pytest.mark.builder
def test_insufficient_candidates_raises(monkeypatch):
    """Fewer valid rows than dataset_target_size ⇒ ValueError."""
    cfg = make_config(dataset_target_size=8, oversample=5)
    b = TranslationDatasetBuilder(cfg)
    monkeypatch.setattr(b, "_load_dataset", lambda: stream_too_few_valid())

    with pytest.raises(ValueError):
        _ = b._extract(b._load_dataset())


@pytest.mark.builder
def test_builder_metrics_counts_on_mixed_stream(monkeypatch):
    """Upfront invalids counted; then collect until target candidates."""
    cfg = make_config(dataset_target_size=5, oversample=2)  # target=10
    b = TranslationDatasetBuilder(cfg)
    monkeypatch.setattr(b, "_load_dataset", stream_known_mix_for_counts)

    out = b._extract(b._load_dataset())

    # 5 invalid first, then 10 valid → seen=15, candidates=10, filtered=5
    assert len(out) == cfg.dataset_target_size
    assert b.builder_metrics.seen == 15
    assert b.builder_metrics.candidates == 10
    assert b.builder_metrics.filtered == 5
    assert b.builder_metrics.filtered_empty == 3
    assert b.builder_metrics.filtered_ratio == 2
    assert b.builder_metrics.selected == cfg.dataset_target_size
