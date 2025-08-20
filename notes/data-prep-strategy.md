# Data Prep Strategy

**1) Sampler (stream → manifest)**

* Does **reservoir sampling** over the HF streaming iterator.
* Applies **pre-filters** (e.g., language keys present, non-empty, char-length caps).
* Emits a **manifest** (e.g., JSONL) with immutable fields per row:

  * `uid` (stable content hash), `src_text`, `tgt_text`, `split` (train/val/test),
  * dataset **revision** / commit id, `seed`, `k`, and any filters used.
* You can re-run to refresh the sample *or* load from disk.

**2) Preprocessor / Tokenizer (manifest → token ids)**

* Takes manifest rows and applies deterministic normalization + **SentencePiece** encode.
* Optionally writes a **tokenized cache** (npz/arrow) so you don’t tokenize every epoch.
* Records tokenizer **artifact hash** (SPM model path + checksum + settings).

**3) Dataset (read-only view)**

* Indexable, returns a single example’s **token ids** (and lengths).
* No padding here—keep examples atomic; it makes caching and testing easier.
* Can be parameterized to read from **local cache** or re-run sampler+preprocess.

**4) Collate / DataLoader**

* **Dynamic padding** to the batch max length (or fixed `T` if you prefer).
* Builds `(tokens, targets, attention_mask)`; for LM, targets = tokens shifted by 1.
* Shuffling, epoch semantics, `drop_last`, **seeded reproducibility**.
* Optional length-based **bucketing** (sort-within-buckets) to reduce pad waste.

# Why not put sampling + tokenization inside `Dataset`?

* **Immutability & reproducibility**: a separate Sampler produces a *frozen* manifest you can version and re-use across experiments.
* **Caching**: tokenization is the slow step; decoupling lets you cache encoded results and swap tokenizers without touching the sampler.
* **Testing**: each stage gets its own invariants (see below), easier to isolate bugs.

# Persistence & reproducibility (what to save)

* `manifest.jsonl` (120 rows) + a tiny `MANIFEST.meta.json`:

  * dataset name, **splits included**, **revision hash**, **seed**, **k**, filter thresholds,
  * reservoir algorithm (“classic”/ES-weighted), timestamp.
* `token_cache/` with encoded arrays and an `ENCODE.meta.json`:

  * SPM model **checksum**, `vocab_size`, `char_coverage`, `byte_fallback`, normalization flags.
* Keep a single **experiment id** tying manifest + encode config together.

# Practical knobs (defaults that work well)

* **Sampling**: `k=120`, `seed=42`, stream `train+validation`, simple char-length cap (e.g., 2–256 chars per side) before tokenization.
* **Tokenization**: SPM Unigram, `vocab=2k–8k`, `byte_fallback=true`. Tokenize **after** sampling.
* **Padding**: dynamic in collate; for your NumPy training, you can also choose a fixed `T` (64–128) and truncate longer examples once at cache time to simplify masks.
* **Batching** (CPU): `B=8–16` is a sweet spot. With 96 train rows, even full-batch is fine.

# Edge cases & failure modes to guard

* **Duplicates**: optional exact-duplicate filter by **normalized text hash** before sampling (or dedup the reservoir before splitting).
* **Empty/degenerate pairs**: drop rows with empty sides or only punctuation.
* **Unicode drift**: fix normalization (NFKC) consistently in both hashing and tokenization.
* **Resampling by accident**: ensure `Dataset` never re-triggers the sampler if a cached manifest is present—make “materialize” an explicit step.
* **Leakage**: split **once** (train/val/test labels in the manifest) and never reshuffle splits downstream.

# Minimal interface sketch (no code, just contracts)

* **Sampler**: `materialize(k, seed, splits, filters) -> manifest_path`
* **Preprocessor**: `encode(manifest_path, spm_model, out_dir) -> cache_dir`
* **Dataset**: `__len__`, `__getitem__(i) -> {ids, length}` (reads cache)
* **DataLoader**: iterator yielding `{tokens: (B,T), targets: (B,T), attn_mask: (B,T,T or B,1,T)}`

# Sanity tests (fast and worth it)

* Reservoir property on synthetic stream (frequency check across many runs).
* Determinism: same `seed+revision` ⇒ identical `uid` set.
* Tokenization parity: re-encode a handful of rows and verify exact match to cache.
* Collate: shapes/dtypes, mask correctness, and loss decreases on a one-batch overfit.

If you follow this split, you keep your PyTorch-like ergonomics, but each layer remains small, testable, and swappable. You’ll also avoid the “mystery resample” and “tokenizer drift” traps that eat time later.

## The Non-OverEngineered Version
totally fair. let’s keep it *mini* and ruthlessly simple.

# the bare-minimum plan (no over-engineering)

1. **sample once (streaming + reservoir)**

   * Iterate `load_dataset("wmt14","fr-en", streaming=True, split="train")`.
   * Keep a 120-item reservoir.
   * After the pass: shuffle once → split to 96/12/12.
   * (Optional) dump to a tiny `manifest.jsonl` so you never resample.

2. **tokenize after you’ve got the 120**

   * Use one tokenizer (e.g., SentencePiece).
   * Encode to ids, then **truncate/pad to a fixed T** (pick 64 or 128).
   * Build targets by shifting by 1. Done.

3. **dataset / loader (keep it dumb)**

   * **Dataset**: just holds 3 numpy arrays: `tokens`, `targets`, `attn_mask` (or build mask on the fly).
   * **Loader**: a tiny iterator that slices those arrays into batches.
   * For 96 training rows, **full-batch** is fine; if you want, use `B=8–16`.

4. **training loop**

   * No epochs gymnastics—just loop a few hundred steps, reshuffling between passes if you like.
   * Skip dropout. AdamW `lr≈1e-3`. Overfit a single batch once to sanity-check.

# what to persist (so you don’t redo work)

* If you care about reuse: save only **the 120 examples** (text), **your seed**, and **tokenizer path**. Everything else can be rebuilt in seconds.

# tiny invariants worth checking (and nothing more)

* Exactly 96/12/12 counts.
* No empty strings.
* Max length ≤ T after tokenization.
* Causal mask blocks future tokens (spot-check one batch).
* One-batch overfit → loss → \~0.

that’s it. if you want, i can sketch *signatures* for the sampler and the super-light dataset/loader (no code dump), but you’ve got the right instincts—keep it boring and tiny.
