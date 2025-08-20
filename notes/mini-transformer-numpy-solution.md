"""
Phase 1 — NumPy-only Mini Transformer
=====================================

A small, educational Transformer implemented from scratch in NumPy with manual
backprop for the critical components: Linear, LayerNorm, single-head Attention,
MLP, Embedding (with weight tying), and Cross-Entropy.

This file is intentionally compact but complete enough to:
- run a forward/backward pass,
- numerically check gradients for a few layers,
- overfit a tiny random dataset (or plug your own integer token dataset).

Design choices
--------------
- Pre-Norm blocks (LayerNorm -> sublayer -> residual)
- Single-head attention for clarity (easy to extend to MHA)
- Learned positional embeddings
- Weight tying between token embedding and LM head
- AdamW optimizer

Note: Dropout omitted initially for simplicity; add after everything works.
"""
from __future__ import annotations
import math
import numpy as np

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def glorot_uniform(shape, rng):
    fan_in, fan_out = shape[0], shape[1]
    limit = math.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def gelu(x):
    # Tanh approximation (Hendrycks & Gimpel)
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * (x ** 3))))


def d_gelu(x):
    # derivative of tanh-approx GELU
    c = np.sqrt(2.0 / np.pi)
    x3 = x ** 3
    tanh_arg = c * (x + 0.044715 * x3)
    t = np.tanh(tanh_arg)
    dt = 1 - t ** 2
    return 0.5 * (1.0 + t) + 0.5 * x * dt * c * (1 + 3 * 0.044715 * x ** 2)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x).astype(np.float32)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + 1e-12)


def causal_mask(T):
    # lower triangular: allow attending to self and previous tokens
    return np.triu(np.ones((T, T), dtype=np.bool_), k=1)


# ------------------------------------------------------------
# Core layers with manual backprop
# ------------------------------------------------------------

class Linear:
    def __init__(self, in_dim, out_dim, rng):
        self.W = glorot_uniform((in_dim, out_dim), rng)
        self.b = np.zeros((out_dim,), dtype=np.float32)
        # grads
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # cache
        self.x = None

    def forward(self, x):
        # x: (N, D)
        self.x = x
        return x @ self.W + self.b

    def backward(self, dout):
        # dout: (N, out_dim)
        self.dW[...] = self.x.T @ dout
        self.db[...] = np.sum(dout, axis=0)
        dx = dout @ self.W.T
        return dx

    def zero_grad(self):
        self.dW.fill(0.0)
        self.db.fill(0.0)


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones((dim,), dtype=np.float32)
        self.beta = np.zeros((dim,), dtype=np.float32)
        self.eps = eps
        # cache
        self.x = None
        self.x_hat = None
        self.mu = None
        self.var = None
        # grads
        self.dgamma = np.zeros_like(self.gamma)
        self.dbeta = np.zeros_like(self.beta)

    def forward(self, x):
        # normalize over last dim
        self.x = x
        self.mu = np.mean(x, axis=-1, keepdims=True)
        self.var = np.var(x, axis=-1, keepdims=True)
        self.x_hat = (x - self.mu) / np.sqrt(self.var + self.eps)
        return self.gamma * self.x_hat + self.beta

    def backward(self, dout):
        # closed-form LN backward (per element)
        self.dbeta[...] = np.sum(dout, axis=tuple(range(dout.ndim - 1)))
        self.dgamma[...] = np.sum(dout * self.x_hat, axis=tuple(range(dout.ndim - 1)))

        N = self.x.shape[-1]
        gamma = self.gamma
        var_eps = np.sqrt(self.var + self.eps)
        g = dout * gamma
        g_mean = np.mean(g, axis=-1, keepdims=True)
        gx_hat_mean = np.mean(g * self.x_hat, axis=-1, keepdims=True)
        dx = (g - g_mean - self.x_hat * gx_hat_mean) / var_eps
        return dx

    def zero_grad(self):
        self.dgamma.fill(0.0)
        self.dbeta.fill(0.0)


class Embedding:
    def __init__(self, vocab_size, dim, rng):
        self.W = (rng.standard_normal((vocab_size, dim)).astype(np.float32) / dim**0.5)
        self.dW = np.zeros_like(self.W)
        self.idx_cache = None

    def forward(self, idx):
        # idx: (B, T) ints
        self.idx_cache = idx
        return self.W[idx]

    def backward(self, dout):
        # dout: (B, T, D)
        np.add.at(self.dW, self.idx_cache, dout)
        return None  # no dx for integer indices

    def zero_grad(self):
        self.dW.fill(0.0)


class PositionalEmbedding:
    def __init__(self, max_T, dim, rng):
        self.W = (rng.standard_normal((max_T, dim)).astype(np.float32) / dim**0.5)
        self.dW = np.zeros_like(self.W)
        self.T_cache = None

    def forward(self, B, T):
        self.T_cache = T
        return self.W[:T][None, :, :].repeat(B, axis=0)

    def backward(self, dout):
        # dout: (B, T, D)
        self.dW[:self.T_cache] += np.sum(dout, axis=0)
        return None

    def zero_grad(self):
        self.dW.fill(0.0)


class SelfAttentionSingleHead:
    def __init__(self, d_model, d_head, rng):
        self.d_model = d_model
        self.d_head = d_head
        # projections
        self.q = Linear(d_model, d_head, rng)
        self.k = Linear(d_model, d_head, rng)
        self.v = Linear(d_model, d_head, rng)
        self.out = Linear(d_head, d_model, rng)
        self.scale = 1.0 / math.sqrt(d_head)
        # caches
        self.X = None
        self.Q = None
        self.K = None
        self.V = None
        self.S = None
        self.P = None
        self.A = None
        self.mask = None

    def forward(self, X):
        # X: (B, T, D)
        B, T, D = X.shape
        self.X = X
        Q = self.q.forward(X.reshape(-1, D)).reshape(B, T, -1)
        K = self.k.forward(X.reshape(-1, D)).reshape(B, T, -1)
        V = self.v.forward(X.reshape(-1, D)).reshape(B, T, -1)
        self.Q, self.K, self.V = Q, K, V

        # scores: (B, T, T)
        S = (Q @ K.transpose(0, 2, 1)) * self.scale
        self.mask = causal_mask(T)
        S = np.where(self.mask[None, :, :], -1e9, S)
        P = softmax(S, axis=-1)
        A = P @ V  # (B, T, d_head)

        self.S, self.P, self.A = S, P, A
        out = self.out.forward(A.reshape(-1, self.d_head)).reshape(B, T, -1)
        return out

    def backward(self, dout):
        B, T, D = self.X.shape
        dA = self.out.backward(dout.reshape(-1, D)).reshape(B, T, self.d_head)

        # A = P @ V
        dP = dA @ self.V.transpose(0, 2, 1)  # (B, T, T)
        dV = self.P.transpose(0, 2, 1) @ dA   # (B, T, d_head)

        # softmax backward row-wise: dS = P * (dP - sum(dP*P))
        sum_dP_P = np.sum(dP * self.P, axis=-1, keepdims=True)
        dS = self.P * (dP - sum_dP_P)

        # apply mask (blocked positions get no gradient)
        dS = np.where(self.mask[None, :, :], 0.0, dS)

        # S = (Q @ K^T) * scale
        dQ = (dS @ self.K) * self.scale
        dK = (dS.transpose(0, 2, 1) @ self.Q) * self.scale

        # back through Q,K,V projections
        dQ_flat = dQ.reshape(-1, self.d_head)
        dK_flat = dK.reshape(-1, self.d_head)
        dV_flat = dV.reshape(-1, self.d_head)

        dX_q = self.q.backward(dQ_flat).reshape(B, T, D)
        dX_k = self.k.backward(dK_flat).reshape(B, T, D)
        dX_v = self.v.backward(dV_flat).reshape(B, T, D)

        dX = dX_q + dX_k + dX_v
        return dX

    def zero_grad(self):
        self.q.zero_grad(); self.k.zero_grad(); self.v.zero_grad(); self.out.zero_grad()


class MLP:
    def __init__(self, d_model, d_hidden, rng):
        self.fc1 = Linear(d_model, d_hidden, rng)
        self.fc2 = Linear(d_hidden, d_model, rng)
        self.x1 = None
        self.h = None

    def forward(self, x):
        self.x1 = self.fc1.forward(x)
        self.h = gelu(self.x1)
        return self.fc2.forward(self.h)

    def backward(self, dout):
        dh = self.fc2.backward(dout)
        dx1 = dh * d_gelu(self.x1)
        dx = self.fc1.backward(dx1)
        return dx

    def zero_grad(self):
        self.fc1.zero_grad(); self.fc2.zero_grad()


class TransformerBlock:
    def __init__(self, d_model, d_head, d_hidden, rng):
        self.ln1 = LayerNorm(d_model)
        self.attn = SelfAttentionSingleHead(d_model, d_head, rng)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model, d_hidden, rng)

    def forward(self, x):
        # Pre-Norm + residual
        x_norm = self.ln1.forward(x)
        a = self.attn.forward(x_norm)
        x = x + a
        x_norm2 = self.ln2.forward(x)
        m = self.mlp.forward(x_norm2)
        x = x + m
        return x

    def backward(self, dout):
        # through second residual
        dm = dout.copy()
        dx = dout.copy()
        # MLP branch
        dx_norm2 = self.mlp.backward(dm)
        dx_ln2 = self.ln2.backward(dx_norm2)
        dx += dx_ln2
        # Attention branch
        da = dx.copy()
        dx_attn = self.attn.backward(da)
        dx_ln1 = self.ln1.backward(dx_attn)
        dx += dx_ln1
        return dx

    def zero_grad(self):
        self.attn.zero_grad(); self.mlp.zero_grad()
        self.ln1.zero_grad(); self.ln2.zero_grad()


# ------------------------------------------------------------
# Model (LM) with weight tying
# ------------------------------------------------------------
class MiniTransformer:
    def __init__(self, vocab_size, max_T, d_model=128, d_head=128, d_hidden=512, n_layers=2, rng=None):
        rng = np.random.default_rng() if rng is None else rng
        self.vocab_size = vocab_size
        self.max_T = max_T
        self.d_model = d_model
        self.tok_emb = Embedding(vocab_size, d_model, rng)
        self.pos_emb = PositionalEmbedding(max_T, d_model, rng)
        self.blocks = [TransformerBlock(d_model, d_head, d_hidden, rng) for _ in range(n_layers)]
        self.ln_f = LayerNorm(d_model)
        # weight tying: decoder head uses tok_emb.W^T
        # No extra parameters needed; gradients will flow into tok_emb.W
        self.cache_xf = None

    def forward(self, idx):
        # idx: (B, T)
        B, T = idx.shape
        assert T <= self.max_T
        x = self.tok_emb.forward(idx) + self.pos_emb.forward(B, T)
        for blk in self.blocks:
            x = blk.forward(x)
        xf = self.ln_f.forward(x)
        self.cache_xf = xf
        # logits = xf @ E^T  (E is token embedding)
        logits = xf @ self.tok_emb.W.T
        return logits

    def backward(self, dlogits):
        # d logits -> d xf, d E
        # logits = xf @ E^T
        # dE accumulates from logits path
        B, T, V = dlogits.shape
        D = self.d_model
        xf = self.cache_xf
        # dE: sum over all (B,T): dlogits[b,t,:]^T * xf[b,t,:]
        self.tok_emb.dW += dlogits.reshape(-1, V).T @ xf.reshape(-1, D)
        dxf = dlogits @ self.tok_emb.W  # (B, T, D)
        # back through final LN and blocks
        dx = self.ln_f.backward(dxf)
        for blk in reversed(self.blocks):
            dx = blk.backward(dx)
        # split gradients to embeddings
        self.pos_emb.backward(dx)
        self.tok_emb.backward(dx)

    def zero_grad(self):
        self.tok_emb.zero_grad(); self.pos_emb.zero_grad(); self.ln_f.zero_grad()
        for blk in self.blocks:
            blk.zero_grad()

    def parameters(self):
        # return list of (param, grad) tuples for optimizer
        params = [(self.tok_emb.W, self.tok_emb.dW), (self.pos_emb.W, self.pos_emb.dW), (self.ln_f.gamma, self.ln_f.dgamma), (self.ln_f.beta, self.ln_f.dbeta)]
        for blk in self.blocks:
            # ln1, ln2
            params += [ (blk.ln1.gamma, blk.ln1.dgamma), (blk.ln1.beta, blk.ln1.dbeta), (blk.ln2.gamma, blk.ln2.dgamma), (blk.ln2.beta, blk.ln2.dbeta) ]
            # attn projections
            params += [ (blk.attn.q.W, blk.attn.q.dW), (blk.attn.q.b, blk.attn.q.db),
                        (blk.attn.k.W, blk.attn.k.dW), (blk.attn.k.b, blk.attn.k.db),
                        (blk.attn.v.W, blk.attn.v.dW), (blk.attn.v.b, blk.attn.v.db),
                        (blk.attn.out.W, blk.attn.out.dW), (blk.attn.out.b, blk.attn.out.db) ]
            # MLP
            params += [ (blk.mlp.fc1.W, blk.mlp.fc1.dW), (blk.mlp.fc1.b, blk.mlp.fc1.db),
                        (blk.mlp.fc2.W, blk.mlp.fc2.dW), (blk.mlp.fc2.b, blk.mlp.fc2.db) ]
        return params


# ------------------------------------------------------------
# Loss: cross entropy with logits (next-token)
# ------------------------------------------------------------

def cross_entropy_logits(logits, targets):
    # logits: (B, T, V); targets: (B, T) ints
    B, T, V = logits.shape
    logits_2d = logits.reshape(-1, V)
    targets_1d = targets.reshape(-1)
    # stable softmax log-prob
    logits_2d = logits_2d - np.max(logits_2d, axis=1, keepdims=True)
    exp = np.exp(logits_2d)
    probs = exp / (np.sum(exp, axis=1, keepdims=True) + 1e-12)
    # loss
    idx = (np.arange(B*T), targets_1d)
    nll = -np.log(probs[idx] + 1e-12)
    loss = np.mean(nll)
    # gradient w.r.t logits
    dlogits = probs
    dlogits[idx] -= 1.0
    dlogits /= (B * T)
    dlogits = dlogits.reshape(B, T, V)
    return loss, dlogits


# ------------------------------------------------------------
# Optimizer: AdamW
# ------------------------------------------------------------
class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-4):
        self.params = params  # list of (param, grad)
        self.lr = lr
        self.b1, self.b2 = betas
        self.eps = eps
        self.wd = weight_decay
        self.m = [np.zeros_like(p) for (p, g) in params]
        self.v = [np.zeros_like(p) for (p, g) in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, (p, g) in enumerate(self.params):
            # weight decay
            if self.wd > 0:
                g = g + self.wd * p
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * (g * g)
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for (p, g) in self.params:
            g.fill(0.0)


# ------------------------------------------------------------
# Gradient check utility
# ------------------------------------------------------------

def grad_check(model, B=2, T=4, V=11, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, V, size=(B, T), dtype=np.int64)
    targets = rng.integers(0, V, size=(B, T), dtype=np.int64)

    # pick a small scalar function of params
    logits = model.forward(idx)
    loss, dlogits = cross_entropy_logits(logits, targets)
    model.zero_grad()
    model.backward(dlogits)

    # test a few random params numerically
    eps = 1e-4
    tol = 5e-3
    params = model.parameters()
    for n in range(5):
        i = rng.integers(0, len(params))
        p, g = params[i]
        # choose a random element
        flat_idx = rng.integers(0, p.size)
        slicer = np.unravel_index(flat_idx, p.shape)
        orig = p[slicer].copy()

        # f(x + eps)
        p[slicer] = orig + eps
        logits_p = model.forward(idx)
        loss_p, _ = cross_entropy_logits(logits_p, targets)

        # f(x - eps)
        p[slicer] = orig - eps
        logits_m = model.forward(idx)
        loss_m, _ = cross_entropy_logits(logits_m, targets)

        # restore
        p[slicer] = orig

        numgrad = (loss_p - loss_m) / (2 * eps)
        backgrad = g[slicer]
        rel_err = abs(numgrad - backgrad) / (abs(numgrad) + abs(backgrad) + 1e-12)
        print(f"Param {i} idx {slicer}: num={numgrad:.5e} back={backgrad:.5e} relerr={rel_err:.3e}")
        assert rel_err < tol, f"Grad check failed: relerr={rel_err} >= {tol}"
    print("Grad check passed for sampled parameters.")


# ------------------------------------------------------------
# Tiny training demo (random data) — replace with your dataset
# ------------------------------------------------------------

def demo_train():
    rng = np.random.default_rng(42)
    V = 2000
    T = 64
    B = 8
    model = MiniTransformer(vocab_size=V, max_T=T, d_model=128, d_head=128, d_hidden=512, n_layers=2, rng=rng)

    # gradient check once on tiny config
    print("Running gradient check on tiny config...")
    tiny = MiniTransformer(vocab_size=11, max_T=8, d_model=16, d_head=16, d_hidden=32, n_layers=1, rng=rng)
    grad_check(tiny, B=2, T=4, V=11, seed=0)

    # synthetic toy data (language modeling style)
    data_tokens = rng.integers(0, V, size=(100, T+1))  # 100 sequences

    optim = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def get_batch():
        idxs = rng.integers(0, data_tokens.shape[0], size=B)
        batch = data_tokens[idxs]
        x = batch[:, :T]
        y = batch[:, 1:T+1]
        return x, y

    print("Training on synthetic data (just to verify plumbing)...")
    for step in range(501):
        x, y = get_batch()
        logits = model.forward(x)
        loss, dlogits = cross_entropy_logits(logits, y)

        model.zero_grad()
        model.backward(dlogits)
        optim.step()

        if step % 50 == 0:
            # compute simple accuracy
            preds = np.argmax(logits, axis=-1)
            acc = (preds == y).mean()
            print(f"step={step:04d} loss={loss:.4f} acc={acc:.3f}")

    print("Done. Replace synthetic data with your tokenized dataset to proceed.")


if __name__ == "__main__":
    demo_train()
