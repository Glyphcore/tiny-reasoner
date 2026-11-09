#!/usr/bin/env python3
"""Thinking-mode direction: mean residual under the thinking chat template minus the non-thinking template, same prompts,
at the last prompt token, for every layer. Saves .cache/inspect-move/dirs-think.pt in the inspect_analyse format so
make_edit.py can bake it (steer) or remove it (orth). Prints one line per 4 layers: |d|/|x| and cosine with the
optimal direction when available."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from tr.inspect import last_token_residuals  # noqa: E402
from tr.policy import default_config, load_policy  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_edit import prompts_from_inspect  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".cache/inspect-move")
    ap.add_argument("--n", type=int, default=48)
    a = ap.parse_args()
    pol = load_policy(default_config(thinking=False, max_new_tokens=8))
    layers = list(range(len(pol.model.model.layers)))
    prompts = prompts_from_inspect(a.dir, a.n)
    rt = last_token_residuals(pol, prompts, layers, enable_thinking=True)
    rn = last_token_residuals(pol, prompts, layers, enable_thinking=False)
    dirs, rows = {}, []
    opt = Path(a.dir) / "dirs-optimal.pt"
    od = torch.load(opt)["dirs"] if opt.exists() else {}
    for L in layers:
        d = rt[L].mean(0) - rn[L].mean(0)
        dirs[L] = d
        rel = float(d.norm() / rn[L].norm(dim=1).mean())
        cos = float(torch.nn.functional.cosine_similarity(d, od[L], dim=0)) if L in od else float("nan")
        # separability is trivially ~1 (template tokens differ); report the within-class spread instead
        spread = float((rt[L] - rt[L].mean(0)).norm(dim=1).mean() / d.norm())
        rows.append((L, rel, cos, spread))
    print(f"n_prompts={len(prompts)}  layer |d|/|x| cos(optimal) spread/|d|")
    for L, rel, cos, spread in rows[::3]:
        print(f"{L:>3} {rel:.3f} {cos:+.2f} {spread:.2f}")
    torch.save({"label": "think", "layers": layers, "dirs": dirs}, Path(a.dir) / "dirs-think.pt")


if __name__ == "__main__":
    main()
