"""Inspection: residual captures, head contributions, logit lens, direction finding. Lead-only analysis code.

All functions take an HFPolicy (model = Qwen3ForCausalLM). Captures are returned on CPU in fp32.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Sequence

from spec.interfaces import Message, Policy


def _layers(policy: Policy):
    return policy.model.model.layers


def _ids(policy: Policy, messages: Sequence[Message], add_generation_prompt=True, enable_thinking=None):
    import torch
    msgs = [{"role": m.role, "content": m.content} for m in messages]
    kw = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    ids = policy.tokenizer.apply_chat_template(msgs, add_generation_prompt=add_generation_prompt, return_tensors="pt", **kw)
    if isinstance(ids, dict) or hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return ids.to(policy.model.device)


def _ids_text(policy: Policy, text: str):
    return policy.tokenizer(text, return_tensors="pt").input_ids.to(policy.model.device)


@contextmanager
def _hooks(modules, fn):
    hs = [m.register_forward_hook(fn(i)) for i, m in enumerate(modules)]
    try:
        yield
    finally:
        for h in hs:
            h.remove()


def residuals(policy: Policy, ids, layers: Sequence[int]):
    """{layer: (seq, d)} residual stream AFTER each listed decoder layer, plus -1 = embeddings."""
    import torch
    out = {}
    mods = _layers(policy)

    def mk(i):
        def hook(mod, inp, res):
            h = res[0] if isinstance(res, tuple) else res
            out[i] = h[0].detach().float().cpu()
        return hook

    want = [mods[i] for i in layers if i >= 0]
    with torch.no_grad(), _hooks(want, lambda j: mk([i for i in layers if i >= 0][j])):
        if -1 in layers:
            out[-1] = policy.model.model.embed_tokens(ids)[0].detach().float().cpu()
        policy.model(ids)
    return out


def head_contributions(policy: Policy, ids, layer: int, head_dim: int = 128):
    """(seq, n_heads, d): what each head of `layer` writes into the residual (o_proj applied per head block)."""
    import torch
    attn = _layers(policy)[layer].self_attn
    got = {}

    def hook(mod, inp, res):
        got["x"] = inp[0][0].detach()  # (seq, n_heads*head_dim), input of o_proj

    h = attn.o_proj.register_forward_hook(hook)
    try:
        with torch.no_grad():
            policy.model(ids)
    finally:
        h.remove()
    x = got["x"].float()
    W = attn.o_proj.weight.detach().float()  # (d, H*hd)
    H = x.shape[1] // head_dim
    xs = x.reshape(x.shape[0], H, head_dim)
    Ws = W.reshape(W.shape[0], H, head_dim)
    return torch.einsum("shk,dhk->shd", xs, Ws).cpu()


def mlp_activations(policy: Policy, ids, layer: int):
    """(seq, 3072) input to down_proj (= act(gate) * up) at `layer`."""
    import torch
    mlp = _layers(policy)[layer].mlp
    got = {}

    def hook(mod, inp, res):
        got["x"] = inp[0][0].detach().float().cpu()

    h = mlp.down_proj.register_forward_hook(hook)
    try:
        with torch.no_grad():
            policy.model(ids)
    finally:
        h.remove()
    return got["x"]


def logit_lens(policy: Policy, h, k: int = 5):
    """Top-k tokens when a residual vector (d,) or (seq, d) is read out through final norm + lm_head."""
    import torch
    m = policy.model
    with torch.no_grad():
        x = h.to(m.device, dtype=m.model.norm.weight.dtype)
        logits = m.lm_head(m.model.norm(x)).float()
        top = logits.topk(k, dim=-1)
    toks = [[policy.tokenizer.decode([int(t)]) for t in row] for row in top.indices.reshape(-1, k)]
    return toks, top.values.reshape(-1, k).cpu()


def diff_of_means(pos, neg):
    """Direction (d,) separating two sets of residual vectors (n, d): mean(pos) - mean(neg), and its unit vector."""
    d = pos.mean(0) - neg.mean(0)
    return d, d / (d.norm() + 1e-8)


def last_token_residuals(policy: Policy, prompts: Sequence[Sequence[Message]], layers: Sequence[int], enable_thinking=None):
    """{layer: (n_prompts, d)} residual at the last prompt token (the position that predicts the first answer token)."""
    import torch
    acc = {l: [] for l in layers}
    for msgs in prompts:
        ids = _ids(policy, msgs, enable_thinking=enable_thinking)
        r = residuals(policy, ids, layers)
        for l in layers:
            acc[l].append(r[l][-1])
    return {l: torch.stack(v) for l, v in acc.items()}


def mean_input(policy: Policy, prompts: Sequence[Sequence[Message]], layer: int, which: str = "o_proj", enable_thinking=None):
    """Mean input vector of o_proj or down_proj at `layer` over all tokens of the prompts (for bake_steer)."""
    import torch
    mod = _layers(policy)[layer]
    target = mod.self_attn.o_proj if which == "o_proj" else mod.mlp.down_proj
    acc = []

    def hook(m, inp, res):
        acc.append(inp[0][0].detach().float().mean(0).cpu())

    h = target.register_forward_hook(hook)
    try:
        with torch.no_grad():
            for msgs in prompts:
                policy.model(_ids(policy, msgs, enable_thinking=enable_thinking))
    finally:
        h.remove()
    return torch.stack(acc).mean(0)
