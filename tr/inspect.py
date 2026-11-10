"""Activation capture on the legality-judgement probes (last prompt position = where the answer word is read out).

python -m tr.inspect capture   -> .cache/acts/{fit,dev}.npz (gitignored): resid, mlp hidden, attention head outputs, logits
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from tr import probes as P
from tr.policy import HFPolicy

ACTS = Path(__file__).resolve().parent.parent / ".cache" / "acts"
SIZES = {"fit": 100, "dev": 50}  # judge probes per game


def answer_tokens(policy):
    t = policy.tok
    return {w: t(w).input_ids[0] for w in ("legal", "illegal", "Legal", "Illegal")}


@torch.no_grad()
def capture(policy, probes):
    layers = policy.model.model.layers
    grab = {"mlp": [None] * len(layers), "head": [None] * len(layers)}
    hooks = []
    for i, layer in enumerate(layers):
        hooks.append(layer.mlp.down_proj.register_forward_pre_hook(
            lambda m, inp, i=i: grab["mlp"].__setitem__(i, inp[0][0, -1].float().cpu())))
        hooks.append(layer.self_attn.o_proj.register_forward_pre_hook(
            lambda m, inp, i=i: grab["head"].__setitem__(i, inp[0][0, -1].float().cpu())))
    toks = answer_tokens(policy)
    out = {k: [] for k in ("resid", "resid_move", "mlp", "head", "logits", "truth", "game", "top")}
    try:
        for p in probes:
            text = policy.render(p.messages)
            ids = policy.tok(text, return_tensors="pt").input_ids.to(policy.device)
            mpos = len(policy.tok(text[:text.index("\nIs this move")]).input_ids) - 1  # last token of the proposed move
            r = policy.model(input_ids=ids, output_hidden_states=True)
            out["resid_move"].append(torch.stack([h[0, mpos].float().cpu() for h in r.hidden_states]).numpy().astype(np.float16))
            lg = r.logits[0, -1].float().cpu()
            out["resid"].append(torch.stack([h[0, -1].float().cpu() for h in r.hidden_states]).numpy().astype(np.float16))
            out["mlp"].append(torch.stack(grab["mlp"]).numpy().astype(np.float16))
            out["head"].append(torch.stack(grab["head"]).numpy().astype(np.float16))
            out["logits"].append([float(lg[toks[w]]) for w in ("legal", "illegal", "Legal", "Illegal")])
            out["top"].append(int(lg.argmax()))
            out["truth"].append(bool(p.meta["truth"]))
            out["game"].append(p.meta["game"])
    finally:
        for h in hooks:
            h.remove()
    return {k: np.array(v) for k, v in out.items()}


def load(split):
    d = np.load(ACTS / f"{split}.npz")
    return {k: d[k] for k in d.files}


def main():
    policy = HFPolicy()
    ACTS.mkdir(parents=True, exist_ok=True)
    for split, n in SIZES.items():
        np.savez(ACTS / f"{split}.npz", **capture(policy, P.judge_probes(split, n)))
        print(f"captured {split} {3 * n}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["capture"]:
        main()
