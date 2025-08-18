# Transformer Model - Mathematics from Encoding to Logits

The **2017 Vaswani et al. encoder–decoder Transformer** written purely as forward-pass math **from embeddings to logits**, using the paper’s **post-norm (“Add & Norm”)** layout.

---

# Notation

* Source length $n$, target length $m$
* Vocab sizes $V_s, V_t$; model dim $d_{\text{model}}$; FF dim $d_{\text{ff}}$
* Heads $h$, per-head dim $d_h=d_{\text{model}}/h$
* Layer counts $L_e$ (encoder), $L_d$ (decoder)
* LayerNorm is $\mathrm{LN}(\cdot)$; Dropout omitted in equations (insert before each Add)
* Masks: $\text{PadMask}$ (to hide padding), $\text{CausalMask}$ (to hide future targets)

---

# Positional + Token Embeddings

Sinusoidal positional encoding (per original paper):

$$
\mathrm{PE}(p,2i)=\sin\!\big(p/10000^{2i/d_{\text{model}}}\big),\quad
\mathrm{PE}(p,2i{+}1)=\cos\!\big(p/10000^{2i/d_{\text{model}}}\big).
$$

**Source (encoder input).** Tokens $s_{1:n}\in\{1,\dots,V_s\}$:

$$
S^{(0)} = E_s[s_{1:n}] + P_{1:n}\;\;\in\mathbb{R}^{n\times d_{\text{model}}},
$$

where $E_s\in\mathbb{R}^{V_s\times d_{\text{model}}}$, $P_{1:n}$ rows are $\mathrm{PE}(p,\cdot)$.

**Target (decoder input).** Tokens $y_{1:m}\in\{1,\dots,V_t\}$:

$$
Y^{(0)} = E_t[y_{1:m}] + P_{1:m}\;\;\in\mathbb{R}^{m\times d_{\text{model}}},
$$

with $E_t\in\mathbb{R}^{V_t\times d_{\text{model}}}$.

---

# Scaled Dot-Product Attention (with masks)

For queries $Q\in\mathbb{R}^{L_q\times d_h}$, keys $K\in\mathbb{R}^{L_k\times d_h}$, values $V\in\mathbb{R}^{L_k\times d_h}$:

$$
\mathrm{Attn}(Q,K,V;\text{Mask})=\mathrm{Softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+\text{Mask}\right)V.
$$

**Masks:** entries are $0$ where allowed, $-\infty$ where disallowed (applied before softmax).

---

# Multi-Head Attention (per head parameters)

For head $i\in\{1,\dots,h\}$ with $W_Q^{(i)},W_K^{(i)},W_V^{(i)}\in\mathbb{R}^{d_{\text{model}}\times d_h}$ and output mix $W_O\in\mathbb{R}^{hd_h\times d_{\text{model}}}$:

$$
\mathrm{MHA}(X;K\!\!=\!X,V\!\!=\!X,\text{Mask})=\mathrm{Concat}(O^{(1)},\dots,O^{(h)})W_O,
$$

$$
Q^{(i)}=X W_Q^{(i)},\quad K^{(i)}=K W_K^{(i)},\quad V^{(i)}=V W_V^{(i)},
$$

$$
O^{(i)}=\mathrm{Attn}\!\big(Q^{(i)},K^{(i)},V^{(i)};\text{Mask}\big).
$$

* **Self-attention:** $K=V=X$.
* **Cross-attention (decoder):** $Q$ from decoder stream, $K,V$ from encoder output.

---

# Position-wise Feed-Forward (shared across positions)

$$
\mathrm{FFN}(x)=\max(0,\,xW_1+b_1)\,W_2+b_2,
$$

with $W_1\in\mathbb{R}^{d_{\text{model}}\times d_{\text{ff}}}$, $W_2\in\mathbb{R}^{d_{\text{ff}}\times d_{\text{model}}}$.

---

# Encoder (post-norm “Add & Norm”)

For layer $\ell=1,\dots,L_e$, input $S^{(\ell-1)}\in\mathbb{R}^{n\times d_{\text{model}}}$:

1. **Self-attention sublayer (no causal mask; apply source PadMask):**

$$
\widetilde{S}=\mathrm{MHA}\!\big(S^{(\ell-1)};\,K{=}S^{(\ell-1)},V{=}S^{(\ell-1)},\,\text{PadMask}_{\text{src}}\big).
$$

$$
U=\mathrm{LN}\!\big(S^{(\ell-1)}+\widetilde{S}\big).
$$

2. **FFN sublayer:**

$$
\widehat{S}=\mathrm{FFN}(U),\quad S^{(\ell)}=\mathrm{LN}\!\big(U+\widehat{S}\big).
$$

**Encoder output:** $H^{\text{enc}}=S^{(L_e)}\in\mathbb{R}^{n\times d_{\text{model}}}$.

---

# Decoder (post-norm; self + cross attention)

For layer $k=1,\dots,L_d$, input $Y^{(k-1)}\in\mathbb{R}^{m\times d_{\text{model}}}$:

1. **Masked self-attention (apply CausalMask + target PadMask):**

$$
\widetilde{Y}_{\text{self}}=\mathrm{MHA}\!\big(Y^{(k-1)};\,K{=}Y^{(k-1)},V{=}Y^{(k-1)},\,\text{CausalMask}\oplus \text{PadMask}_{\text{tgt}}\big),
$$

$$
U=\mathrm{LN}\!\big(Y^{(k-1)}+\widetilde{Y}_{\text{self}}\big).
$$

2. **Cross-attention (queries from decoder, keys/values from encoder; apply source PadMask broadcast to target length):**

$$
\widetilde{Y}_{\text{cross}}=\mathrm{MHA}\!\big(U;\,K{=}H^{\text{enc}},V{=}H^{\text{enc}},\,\text{PadMask}_{\text{src}\rightarrow\text{tgt}}\big),
$$

$$
V'=\mathrm{LN}\!\big(U+\widetilde{Y}_{\text{cross}}\big).
$$

3. **FFN sublayer:**

$$
\widehat{Y}=\mathrm{FFN}(V'),\quad Y^{(k)}=\mathrm{LN}\!\big(V'+\widehat{Y}\big).
$$

**Decoder output:** $H^{\text{dec}}=Y^{(L_d)}\in\mathbb{R}^{m\times d_{\text{model}}}$.

---

# Projection to Logits (per target position)

Optionally tie weights $W_{\text{LM}} = E_t^\top$. In general:

$$
\text{logits} = H^{\text{dec}} W_{\text{LM}} + b \;\;\in \mathbb{R}^{m\times V_t},
$$

with $W_{\text{LM}}\in\mathbb{R}^{d_{\text{model}}\times V_t}$, $b\in\mathbb{R}^{V_t}$.

(If you want probabilities: $\;p(y_t\!\mid\!\cdot)=\mathrm{Softmax}(\text{logits}_t)$.)

---

## Shapes (per layer, per stream)

* $S^{(\ell)},H^{\text{enc}}$: $(n, d_{\text{model}})$
* $Y^{(k)},H^{\text{dec}}$: $(m, d_{\text{model}})$
* Per-head $Q,K,V$: $(n\text{ or }m, d_h)$; concatenated MHA output: $(n\text{ or }m, d_{\text{model}})$
* Logits: $(m, V_t)$

---

That’s the complete **embeddings → logits** path for the original encoder–decoder Transformer, faithful to the 2017 post-norm design, with masking where it belongs. If you want, I can mirror these symbols into a PyTorch skeleton so you can code straight from the equations and line-up shapes as you go.
