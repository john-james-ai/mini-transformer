# CPU Friendly Architecture
A desktop CPU can comfortably train a **mini-Transformer** for toy MT or next-token LM, especially if your goal is to *grok the training mechanics* rather than chase SOTA.

# What’s feasible on CPU (no big GPU)

Below are CPU-friendly configs (weight tying on, learned pos-emb, fp32, batch=8, seq=128). Numbers are order-of-magnitude so you can sanity-check.

| Config    | Layers | d\_model | d\_ff | Vocab |  Params | Param mem | Extra (grads+Adam) | Train FLOPs/step\* |
| --------- | -----: | -------: | ----: | ----: | ------: | --------: | -----------------: | -----------------: |
| **Nano**  |      2 |      128 |   512 |    2k | \~0.67M |  \~2.7 MB |           \~8.1 MB |       \~2.8 GFLOPs |
| **Micro** |      4 |      256 |  1024 |    4k | \~4.28M | \~17.1 MB |          \~51.3 MB |      \~20.9 GFLOPs |
| **Mini**  |      6 |      384 |  1536 |    8k | \~13.8M | \~55.3 MB |         \~165.9 MB |      \~68.9 GFLOPs |

\*Per step at batch=8, seq=128. Activation memory is also modest at these sizes (≈6 MB, 25 MB, 57 MB respectively, rough).

On a modern 8–16-core desktop, those GFLOP budgets are totally doable with MKL/Accelerate/oneDNN. Expect “minutes, not hours” to overfit 50–100 examples with **Nano/Micro**. The **Mini** tier is still fine if you keep sequences short.

# Make it CPU-friendly

* **Keep T small**: attention is $O(T^2)$. Start with `T=64–128`.
* **Small vocab**: char-level or a **2–8k** BPE. Weight-tied LM head.
* **Tiny model**: 1–4 layers, `d_model=128–256`, MLP `4×d_model`.
* **No dropout at first**: it just slows learning on tiny data.
* **AdamW** `lr≈1e-3`, warmup \~200 steps, clip grad=1.0.
* **Overfit one batch first**: prove plumbing → then scale.
* **Batching**: gradient accumulate if you need “bigger” batches.
* **Numerics**: stable softmax (`x - x.max(...)`), LayerNorm `ε=1e-5`.

# NumPy vs PyTorch (on a CPU)

* **NumPy**: maximal learning—write forward *and* backward for Linear, Softmax, Cross-Entropy, LayerNorm, Attention. Use `einsum`/batched matmuls (avoid Python loops). You can add **Numba** later for speed, but plain NumPy with BLAS is fine at Nano/Micro scale.
* **PyTorch (CPU)**: faster and less boilerplate. Still “grok” it by building blocks from `torch.*` primitives (avoid `nn.MultiheadAttention`/`nn.LayerNorm` initially); compare autograd grads to your hand-rolled NumPy grads for the same tensors. You’ll iterate far quicker once the math is verified.

# When you’ll *want* a GPU (or Apple M-series)

* You push **T ≥ 512**, **layers ≥ 8**, **d\_model ≥ 512**, or **dataset ≥ 100k tokens** and want reasonable wall-clock.
* You’re exploring scheduling/regularization sweeps or larger vocabularies.
  (Apple Silicon’s **MPS** backend gives you a nice middle ground if you’re on a Mac.)

# A minimal training recipe to overfit 50–100 pairs

1. **Config**: 2 layers, `d_model=128`, `heads=4`, `d_ff=512`, `vocab=2k`, `seq=64–128`.
2. **Data**: make contiguous chunks (teacher forcing), pad/truncate to `T`.
3. **Train**: AdamW `lr=1e-3`, warmup 200, no dropout, weight decay `1e-4`.
4. **Targets**: next-token cross-entropy; report loss+accuracy per 100 steps.
5. **Checks**: gradient check tiny shapes; unit-test mask logic; overfit one batch to near-zero loss.

If you want, I can drop a concrete file layout plus param/FLOPs calculators and a tiny gradient-check harness so you can flip between NumPy and PyTorch on the same shapes.
