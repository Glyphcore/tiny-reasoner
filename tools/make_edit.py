#!/usr/bin/env python3
"""Build a RecipeEdit against the live base model, check apply + exact revert, save to edits/<id>.{json,safetensors}.

Examples
  tools/make_edit.py taskvec --id tv-0.3 --lam 0.3
  tools/make_edit.py steer --id steer-L14-2 --layer 14 --where o_proj --dirs .cache/inspect/dirs-optimal.pt --scale 2.0
  tools/make_edit.py orth --id orth-L10-20 --layers 10-20 --dirs .cache/inspect/dirs-optimal.pt --dir-layer 14 --negate
  tools/make_edit.py headscale --id hs-L5-3 --layer 5 --heads 3,7 --factor 0.0
Prints one line: id, kind, n tensors, max relative change, revert ok. Leaves the model reverted.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from spec.interfaces import Message  # noqa: E402
from tr.edits.recipes import RecipeEdit  # noqa: E402
from tr.policy import default_config, load_policy  # noqa: E402

BASE_OTHER = ("Qwen/Qwen3-0.6B-Base", "da87bfb608c14b7cf20ba1ce41287e8de496c0cd")


def span(s):
    lo, _, hi = s.partition("-")
    return list(range(int(lo), int(hi or lo) + 1))


def prompts_from_inspect(path=".cache/inspect", n=24):
    """A few real game prompts (from the collected states) for mean-input estimation."""
    from tr.episode import GameEpisode
    from tr.games import get_game
    out = []
    for g in ["hanoi", "puzzle8", "countdown"]:
        p = Path(path) / f"{g}.pt"
        if not p.exists():
            continue
        d = torch.load(p)
        game, ver = get_game(g)
        ep = GameEpisode(game, ver, None)
        for m in d["meta"][:: max(1, len(d["meta"]) // (n // 3))][: n // 3]:
            s = game.decode(m["state"])
            out.append([Message("system", ep.system()), Message("user", ep.solo_prompt(s))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["taskvec", "steer", "orth", "headscale", "neuron"])
    ap.add_argument("--id", required=True)
    ap.add_argument("--lam", type=float)
    ap.add_argument("--include", default="self_attn,mlp")
    ap.add_argument("--layer", type=int)
    ap.add_argument("--layers")
    ap.add_argument("--where", default="o_proj")
    ap.add_argument("--dirs", help=".pt from inspect_analyse (dict layer -> direction)")
    ap.add_argument("--dir-layer", type=int, help="layer whose direction to use (default --layer)")
    ap.add_argument("--negate", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--heads")
    ap.add_argument("--idx")
    ap.add_argument("--factor", type=float, default=0.0)
    ap.add_argument("--out", default="edits")
    a = ap.parse_args()

    pol = load_policy(default_config(thinking=False, max_new_tokens=8))
    aux = {}
    if a.kind == "taskvec":
        recipe = {"kind": "taskvec", "other": BASE_OTHER[0], "other_revision": BASE_OTHER[1], "lam": a.lam,
                  "include": a.include.split(",")}
    else:
        d = None
        if a.dirs:
            dl = a.dir_layer if a.dir_layer is not None else a.layer
            d = torch.load(a.dirs)["dirs"][dl].float()
            d = d / d.norm()
            if a.negate:
                d = -d
            aux["dir"] = d
        if a.kind == "steer":
            from tr.inspect import mean_input
            m = mean_input(pol, prompts_from_inspect(), a.layer, a.where, enable_thinking=False)
            aux["mean_input"] = m
            # scale is in units of the typical residual norm at that layer: resid norm ~ |x| measured on the same prompts
            from tr.inspect import last_token_residuals
            xn = last_token_residuals(pol, prompts_from_inspect(n=9), [a.layer], enable_thinking=False)[a.layer].norm(dim=1).mean()
            recipe = {"kind": "steer", "layer": a.layer, "where": a.where, "dir": "dir", "mean_input": "mean_input",
                      "scale": float(a.scale * xn), "scale_rel": a.scale, "resid_norm": float(xn)}
        elif a.kind == "orth":
            recipe = {"kind": "orth", "layers": span(a.layers), "dir": "dir"}
        elif a.kind == "headscale":
            recipe = {"kind": "headscale", "layer": a.layer, "heads": [int(h) for h in a.heads.split(",")], "factor": a.factor}
        else:
            recipe = {"kind": "neuron", "layer": a.layer, "idx": [int(i) for i in a.idx.split(",")], "factor": a.factor}

    edit = RecipeEdit(a.id, recipe, aux)
    before = {n: pol.tensor(n).detach().float().norm().item() for n in edit._targets(pol)} if a.kind != "taskvec" else {}
    edit.apply(pol)
    rel = 0.0
    for n, b in before.items():
        rel = max(rel, (pol.tensor(n).detach().float().norm().item() - b) / b)
    ok_after = edit.verify(pol, "after")
    edit.revert(pol)
    ok_before = edit.verify(pol, "before")
    edit.save(f"{a.out}/{a.id}")
    print(f"{a.id} kind={edit.kind} tensors={len(edit.changes)} max_rel_norm_change={rel:.3f} after_ok={ok_after} revert_ok={ok_before}")


if __name__ == "__main__":
    main()
