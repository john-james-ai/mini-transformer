# Data Prep Strategy: Numpy vs Pytorch

## ✅ Good to outsource

**Tokenization** (text → IDs)

* It’s I/O and string processing, not differentiable.
* Using Hugging Face `tokenizers` / SentencePiece is fine. You’ll feed **integer ID arrays** into your NumPy model.

## 🚫 Don’t outsource if you want to train in NumPy

**Embedding layer** (IDs → vectors) and **positional encoding** (add order)

* If you compute embeddings in PyTorch and the rest in NumPy, you **cannot backprop into the Torch embedding** from NumPy. It will be **frozen**.
* For learning, you want the gradients to update the **embedding matrix** (and everything downstream). That’s trivial in NumPy for a toy model.
* Positional encodings are just math; implement in NumPy in minutes.

# Two viable setups

## Option A — Recommended for learning

* **Tokenizer**: HF/SentencePiece (Python API) → returns `np.int32` IDs.
* **Everything else**: NumPy (Embedding params, PosEnc, QKV, attention, FFN, logits, loss, your own backprop/updates).

Pros: end-to-end gradients in one framework; you truly “grok training.”

## Option B — If you insist on mixing

* **Tokenizer**: HF/SentencePiece.
* **Embeddings/PE**: PyTorch (forward only), **frozen**. Export each batch to NumPy with `.detach().cpu().numpy()` and continue the model in NumPy.

Cons: you lose learning signal for embeddings; extra copies across frameworks; easy to make dtype/shape mistakes. Only useful if you *explicitly* want fixed embeddings.

# What you actually need to implement in NumPy (it’s not that bad)

## 1) Embedding layer (trainable)

```python
# params
E = 0.02 * np.random.randn(V, d).astype(np.float32)  # vocab, d_model

# forward: ids -> vectors
def embed(ids):  # ids: (B, T) int
    return E[ids]  # (B, T, d)
```

Backprop: accumulate gradients for the rows you indexed (simple scatter-add).

## 2) Positional encoding (sinusoidal)

```python
def sinusoid_pe(T, d):
    pos = np.arange(T)[:, None]
    i = np.arange(d)[None, :]
    theta = pos / (10000 ** ( (2*(i//2)) / d ))
    pe = np.zeros((T, d), dtype=np.float32)
    pe[:, 0::2] = np.sin(theta[:, 0::2])
    pe[:, 1::2] = np.cos(theta[:, 1::2])
    return pe  # (T, d)

# use:
PE_src = sinusoid_pe(S_max, d)  # broadcast to (B, T, d)
PE_tgt = sinusoid_pe(T_max, d)
```

## 3) Masked self-attention & cross-attention

* Keep consistent shapes: `(B, T, d)` → project to `(B, h, T, d_h)`.
* Use a **causal mask** for decoder self-attn and a **pad mask** for both streams.
* Add mask as `-1e9` before softmax.

## 4) Loss & training loop

* Cross-entropy on logits vs target IDs (ignore pad).
* Optimizer: start with SGD or Adam you write yourself (single file).
* Manual backprop: do-able for a 1–2 layer toy. If that’s too heavy, start with a **bigram LM** to warm up, then add attention.

# Interface contract: mixing tokenizers with your NumPy model

**Tokenizer side (HF tokenizers / SentencePiece)**

```python
# Example with HF tokenizers
ids_src = tokenizer_src.encode_batch(list_of_src_strings)
ids_tgt = tokenizer_tgt.encode_batch(list_of_tgt_strings)
# pad to (B, S_max) and (B, T_max), build masks and shifted Y_in / labels Y_out
X_ids = np.array(padded_src_ids, dtype=np.int32)     # (B, S_max)
Y_in  = np.array(padded_tgt_ids_shifted, np.int32)   # (B, T_max)
Y_out = np.array(padded_tgt_labels, np.int32)        # (B, T_max)
M_src = (X_ids != pad_id).astype(np.float32)         # (B, S_max)
M_tgt = (Y_out != pad_id).astype(np.float32)         # (B, T_max)
```

**NumPy model expects**:

* `X_ids, Y_in, Y_out, M_src, M_tgt`
* It does: `X = embed(X_ids) + PE_src[:S_max]; Y0 = embed(Y_in) + PE_tgt[:T_max]`
* Then decoder: masked self-attn → cross-attn with encoder → FFN → logits → CE loss.

# Practical advice to avoid pain

* **Keep everything on CPU float32**. Mixed dtypes or float64 will bite you in softmax and Adam.
* **Freeze vocab early**. Tokenizer must match your embedding matrix `V`. Export `pad_id`, `bos_id`, `eos_id`.
* **Batch by similar lengths** (bucketing) to reduce padding and speed up.
* **Start tiny**: `d=32`, `h=2`, `L=1`, vocab < 200, seq len ≤ 8, batch 8–16. Prove training drops loss and reaches >95% accuracy on your tiny set; then add depth/width.
* **Unit test blocks**: write small shape/gradient checks for: embeddings, softmax+mask, attention weights sum to \~1 where unmasked, CE ignores pads.

# A minimal plan (weekend-scale)

1. **Data**: make 20–50 parallel pairs (or use a tiny Tatoeba sample), tokenize with HF tokenizers → NumPy IDs.
2. **Model**: implement Embedding, Sinusoid PE, 1-layer encoder, 1-layer decoder, attention + FFN.
3. **Train**: write CE loss + simple SGD/Adam; train until it outputs the correct English for the few training pairs (i.e., “memorize” → confirms your gradients).
4. **Infer**: greedy decode; verify it produces the right strings.

If you want, I can give you:

* a **tiny tokenizer stub** (pure-Python “split on space” with a toy vocab) to eliminate external deps, or
* a **NumPy training skeleton** (functions and parameter dicts), into which you drop the tokenized arrays.

Tell me which you prefer and I’ll spit out a ready-to-run scaffold.
