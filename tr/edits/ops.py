"""Computed weight operations (fp32 in, fp32 out). Each returns a new tensor and is used through make_edit.

Qwen3 layout (hidden d=1024, 16 q heads x 128, 8 kv heads, mlp 3072):
  self_attn.o_proj.weight  (d, n_heads*head_dim)   columns [h*128:(h+1)*128] = head h writes into the residual
  mlp.down_proj.weight     (d, 3072)                column j = neuron j writes into the residual
  mlp.up_proj / gate_proj  (3072, d)                row j = neuron j reads from the residual
  embed_tokens / lm_head   (vocab, d)
Weights that WRITE into the residual have the residual dimension on rows (dim 0); ones that READ have it on columns (dim 1).
"""
from __future__ import annotations

from typing import Sequence


def unit(v):
    return v / (v.norm() + 1e-8)


def orthogonalize_rows(W, v):
    """Remove direction v from everything W writes into the residual: W - v v^T W (W is (d, in), v is (d,))."""
    v = unit(v.to(W)).reshape(-1, 1)
    return W - v @ (v.T @ W)


def orthogonalize_cols(W, v):
    """Remove direction v from what W reads: W - (W v) v^T (W is (out, d), v is (d,))."""
    v = unit(v.to(W)).reshape(-1, 1)
    return W - (W @ v) @ v.T


def add_rank1(W, u, v, scale=1.0):
    """W + scale * u v^T. For o_proj/down_proj: u is the residual direction to write, v the input pattern that triggers it."""
    return W + scale * u.to(W).reshape(-1, 1) @ v.to(W).reshape(1, -1)


def bake_steer(W, direction, mean_input, scale):
    """Fold a constant steering vector into a writing weight: on inputs x with x . m_hat ~ |m|, adds scale*direction.

    delta = scale * direction (x) m / |m|^2, so delta @ m = scale * direction. Exact on the mean input, approximate elsewhere."""
    m = mean_input.to(W)
    return add_rank1(W, direction, m / (m.norm() ** 2 + 1e-8), scale)


def scale_heads(W_o, heads: Sequence[int], factor: float, head_dim: int = 128):
    W = W_o.clone()
    for h in heads:
        W[:, h * head_dim:(h + 1) * head_dim] *= factor
    return W


def scale_cols(W, idx: Sequence[int], factor: float):
    W = W.clone()
    W[:, list(idx)] *= factor
    return W


def scale_rows(W, idx: Sequence[int], factor: float):
    W = W.clone()
    W[list(idx), :] *= factor
    return W


def task_vector(W_base, W_other, scale: float):
    """W_base + scale * (W_other - W_base): arithmetic between released checkpoints with the same shapes."""
    return W_base + scale * (W_other.to(W_base) - W_base)


def low_rank(delta, rank: int):
    """SVD-truncate a computed delta to LoRA form: returns (A (r, in), B (out, r)) with B @ A ~ delta."""
    import torch
    U, S, Vh = torch.linalg.svd(delta.float(), full_matrices=False)
    r = min(rank, S.numel())
    B = U[:, :r] * S[:r]
    A = Vh[:r, :]
    return A, B
