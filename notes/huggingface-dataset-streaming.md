# HuggingFace Dataset Streaming
Here’s the “how” and the “why,” plus a minimal pattern you can adapt without me doing the whole thing for you.

# What streaming gives you (HF Datasets)

* `load_dataset(..., streaming=True)` returns an **IterableDataset** (no random access, no `.select`, no exact `.shuffle()` like arrow-mode).
* You iterate once over the corpus—**no full download / no materialization**—perfect for tiny, uniform sampling.

# Why reservoir sampling here

* When the dataset length is unknown (or you don’t want to load it), **reservoir sampling** gives an **exactly uniform** sample of size `k` from a single pass, independent of stream order.
* Space/time: **O(k) memory, O(N) time**, one pass.

# Minimal pattern (adapt it)

```python
from datasets import load_dataset
import random

def reservoir_sample(iterable, k, seed=42):
    rng = random.Random(seed)
    R = []
    for i, ex in enumerate(iterable, 1):  # 1-based index
        if i <= k:
            R.append(ex)
        else:
            j = rng.randrange(i)  # 0..i-1
            if j < k:
                R[j] = ex
    return R

# 1) Stream WMT14 En–Fr (use train, or "train+validation" if you want a bigger pool)
stream = load_dataset("wmt14", "fr-en", split="train", streaming=True)

# 2) Uniformly sample 120 examples in one pass
reservoir = reservoir_sample(stream, k=120, seed=42)   # list of dicts

# 3) Deterministically split 80/10/10
rng = random.Random(42)
rng.shuffle(reservoir)
train, val, test = reservoir[:96], reservoir[96:108], reservoir[108:]
```

Each element looks like:

```python
ex["translation"]["en"], ex["translation"]["fr"]
```

# Notes & gotchas (so you don’t trip later)

* **Split composition**: You can stream multiple splits at once with `split="train+validation"`. Reservoir sampling remains uniform over the combined stream.
* **Determinism**: Fix `seed`, and **pin the dataset revision/version** if you need strict reproducibility (datasets can update URLs/mirrors over time).
* **Do *not* mix with buffer shuffles** if you want strict uniformity. (IterableDataset’s buffer shuffles approximate randomness but aren’t uniform w\.r.t. the full corpus.)
* **Filtering**: If you filter on the fly (e.g., drop very long sentences), do it **before** the reservoir step; the sample is then uniform over the filtered stream.
* **Throughput**: Tokenization-on-the-fly will dominate time. If you’re going to tokenize, consider doing it **after** you’ve selected the 120 to keep things fast.
* **Language direction**: The config name is `"fr-en"`, but you can swap source/target at use-time. If you truly need the reverse config, HF also accepts `"en-fr"` for some WMT years; otherwise just flip fields.

# Quick sanity checks (worth doing)

* Print a few examples to confirm shape/content.
* Confirm `len(train)=96, len(val)=12, len(test)=12`.
* If you care about sentence length distribution, compute it on the reservoir and ensure it matches expectations (uniformity is in *identity*, not by length).

If you want, I can sketch a tiny test harness to verify uniformity (e.g., simulate multiple runs on a synthetic stream and check selection frequencies), but I’ll wait for your signal.
