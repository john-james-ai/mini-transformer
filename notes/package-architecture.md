# Mini-Transformer Package Architecture
Here’s a clean, incremental **project layout** for a NumPy-only mini-Transformer that’s easy to build, test, and extend. It keeps tokenization simple (pure Python), isolates the math, and lets you bolt on training later without refactors.

---

# 📁 Package layout

```
mini_transformer/
├─ pyproject.toml            # or setup.cfg — minimal packaging
├─ README.md                 # how to run, goals, design notes
├─ LICENSE
├─ mini_transformer/
│  ├─ __init__.py
│  ├─ config.py              # dataclasses / config loading
│  ├─ data/
│  │  ├─ __init__.py
│  │  ├─ datasets.py         # load pairs, split, batching, padding
│  │  ├─ tokenization.py     # toy tokenizer, vocab build, encode/decode
│  │  └─ io.py               # read HF sample, csv/tsv loaders (optional)
│  ├─ nn/
│  │  ├─ __init__.py
│  │  ├─ layers.py           # linear, embeddings, gelu, softmax, layernorm
│  │  ├─ attention.py        # (masked) self-attn, cross-attn
│  │  ├─ transformer.py      # Encoder, Decoder, Seq2Seq wrapper
│  │  └─ losses.py           # masked cross-entropy
│  ├─ train/
│  │  ├─ __init__.py
│  │  ├─ optim.py            # SGD/Adam (NumPy)
│  │  ├─ loop.py             # forward/backward/update steps
│  │  └─ metrics.py          # loss trackers, accuracy on non-pad
│  ├─ utils/
│  │  ├─ __init__.py
│  │  ├─ mask.py             # causal & pad masks
│  │  ├─ init.py             # RNG helpers, Xavier/normal inits
│  │  └─ logging.py          # simple tqdm/print wrappers
│  └─ cli/
│     ├─ __init__.py
│     ├─ prepare.py          # build vocab, save artifacts
│     ├─ train.py            # run training from config
│     └─ translate.py        # greedy decode from a saved checkpoint
├─ experiments/
│  ├─ configs/
│  │  ├─ tiny_enfr.yaml      # d=32,h=1,L=1,vocabs small
│  │  └─ small_enfr.yaml     # d=64,h=2,L=2
│  ├─ notebooks/
│  │  ├─ 01_sanity_forward.ipynb   # run 1 batch forward only
│  │  └─ 02_train_tiny.ipynb       # train to memorize 50 pairs
│  └─ runs/                   # saved checkpoints, logs (gitignored)
└─ tests/
   ├─ test_tokenization.py
   ├─ test_masks.py
   ├─ test_attention_shapes.py
   ├─ test_losses.py
   └─ test_forward_decode.py
```

---

# 🔧 Key modules & responsibilities

## `data/tokenization.py`

* Build vocabulary with specials: `<pad>=0,<bos>=1,<eos>=2>`.
* Pure-Python whitespace tokenizer (deterministic).
* `encode(str) -> List[int]`, `decode(List[int]) -> str`.
* `build_vocab(pairs, min_freq=1) -> (w2i, i2w)`.

**Signatures**

```python
def build_vocab(pairs, specials=("<pad>", "<bos>", "<eos>")) -> dict:
    ...

def encode(text: str, w2i: dict) -> list[int]:
    ...

def decode(ids: list[int], i2w: dict) -> str:
    ...
```

## `data/datasets.py`

* Split into train/val.
* Batchify with **padding** and **shifted targets**.
* Returns `X_ids, M_src, Y_in, Y_out, M_tgt`.

```python
def make_batch(pairs, w2i_src, w2i_tgt, batch_size: int) -> dict[str, np.ndarray]:
    # returns dict with X_ids, M_src, Y_in, Y_out, M_tgt
```

## `utils/mask.py`

* `causal(T) -> (T,T)` upper-tri mask with `-1e9`.
* `key_pad(mask_bS) -> broadcastable (B,1,S)` `-1e9` where pads.

## `nn/layers.py`

* `Embedding(V,d)` with gather/scatter-grad.
* `linear(x, W, b)` + backprop helpers.
* `gelu`, `softmax_stable`, `layernorm` (add later if you want).

## `nn/attention.py`

* **Single-head first** (extend to multi-head later).
* `self_attention(Q,K,V,mask)` → `(context, attn_weights)`
* Pack QKV projections inside `decoder_self_attn` and `cross_attn`.

## `nn/transformer.py`

* `EncoderLayer`, `DecoderLayer`, `Encoder`, `Decoder`, `Seq2Seq`.
* Keep **pre-norm optional**; start without layernorm.

**Core forward contract**

```python
def encoder_forward(X_ids, M_src, params, pe_src) -> np.ndarray:  # (B,S,d)
    ...

def decoder_forward(Y_in, M_tgt, Henc, M_src, params, pe_tgt) -> np.ndarray:  # (B,T,d)
    ...

def logits_forward(Hdec, params) -> np.ndarray:  # (B,T,V_t)
    ...
```

## `nn/losses.py`

* Masked token-level cross entropy.

```python
def cross_entropy_masked(logits, y_true, mask) -> tuple[float, np.ndarray]:
    # returns (loss_scalar, dlogits)
```

## `train/optim.py`

* Minimal **SGD** and **Adam** (NumPy dict of params → dict of grads).

```python
class Adam:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8): ...
    def step(self, grads): ...
```

## `train/loop.py`

* One `step()` that: forward → loss → backward → update.
* Epoch loop with train/val loss print.

```python
def train_epoch(model, data_iter, params, opt, pe_src, pe_tgt):
    for batch in data_iter:  # yields dicts from datasets.make_batch
        loss = step(batch, params, opt, pe_src, pe_tgt)
```

## `cli/prepare.py`

* Load 50–100 pairs (your HF sample).
* Build vocabs, save `*.json` for `w2i/i2w`.
* Serialize tokenized pairs for fast reload (`.npz`).

## `cli/train.py`

* Read `experiments/configs/*.yaml`.
* Build params dict with shapes from config.
* Train, save `runs/<exp>/checkpoint.npz`.

## `cli/translate.py`

* Load checkpoint + vocabs.
* Greedy decode: source string → target string.

---

# 🧩 Config (YAML) example

`experiments/configs/tiny_enfr.yaml`

```yaml
seed: 0
data:
  train_path: data/enfr_small.csv
  val_split: 0.1
  lowercase: true
  max_samples: 100
  batch_size: 16
model:
  d_model: 32
  d_ff: 64
  n_layers_enc: 1
  n_layers_dec: 1
  n_heads: 1        # start single-head; expand later
  tie_embeddings: false
train:
  epochs: 30
  lr: 0.001
  optimizer: adam
  clip_grad_norm: 1.0
  print_every: 50
special_tokens:
  pad: "<pad>"
  bos: "<bos>"
  eos: "<eos>"
```

---

# ▶️ Minimal build order (so you see progress fast)

1. **Tokenization pipeline** (`data/tokenization.py`)

   * Build `w2i/i2w`, `encode/decode`. Unit tests.

2. **Batching with masks** (`data/datasets.py`, `utils/mask.py`)

   * Produce `X_ids, M_src, Y_in, Y_out, M_tgt` for a batch. Test shapes.

3. **Embeddings + PE + linear** (`nn/layers.py`)

   * Forward only; assert shapes.

4. **Self-attention (single-head)** (`nn/attention.py`)

   * Implement attention & masking. Unit test: rows of softmax sum ≈ 1 on non-masked.

5. **Encoder/Decoder forward** (`nn/transformer.py`)

   * 1 layer each. Forward a batch; no NaNs.

6. **Loss** (`nn/losses.py`)

   * Cross-entropy masked. Confirm loss \~log(V\_t) at init.

7. **Backward passes** (in each layer file)

   * Start from `dlogits` → backprop through decoder (FFN → cross → self) → encoder.
   * Verify gradient shapes; add tiny numeric grad checks on small tensors.

8. **Optimizer + loop** (`train/optim.py`, `train/loop.py`)

   * Train on 50–100 pairs until train loss ↓ and greedy decode gives correct outputs.

9. **CLI** (`cli/train.py`, `cli/translate.py`)

   * End-to-end: `python -m mini_transformer.cli.train --config experiments/configs/tiny_enfr.yaml`

---

# 🧪 Tests you actually want

* `test_tokenization.py`: specials present; encode/decode round trips.
* `test_masks.py`: causal mask upper-tri; pad mask zeros out the right positions.
* `test_attention_shapes.py`: Q/K/V and scores shapes; softmax rows sum to \~1 where allowed.
* `test_losses.py`: CE equals manual computation on a tiny batch.
* `test_forward_decode.py`: with **fixed tiny params**, forward runs and greedy decode returns a sequence of valid IDs (ends with `<eos>`).

---

# 📝 API sketch (for your editor)

```python
# mini_transformer/nn/transformer.py
class Seq2Seq:
    def __init__(self, params): ...
    def forward(self, X_ids, M_src, Y_in, M_tgt):  # returns logits
        ...
    def greedy_decode(self, X_ids, M_src, max_T=16):
        ...
```

---

# 🔄 Dependency stance

* **NumPy only** (+ `pyyaml` for configs if you like).
* No PyTorch/TF; no HF tokenizer dependency (you can add later).
* Keep arrays `float32`; IDs `int32`.

---

If you want, I can draft **empty file stubs with function signatures** for each module so you can fill in the math without wrestling with structure.
