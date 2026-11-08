#!/usr/bin/env python3
"""Rank attention heads by how differently they write at the move position for optimal vs wrong moves.

For each collected state (meta from inspect_collect --at move) we rebuild the prompt + reply prefix, take each head's
contribution to the residual at the last position (tr.inspect.head_contributions) and project it on the per-layer
optimal direction (dirs-<label>.pt from inspect_analyse). Score(layer, head) = mean projection on optimal states minus
mean on wrong states, in units of the pooled std. Prints the top and bottom 12 heads; saves .cache/.../heads-<label>.pt."""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from spec.interfaces import Message  # noqa: E402
from tr.episode import GameEpisode  # noqa: E402
from tr.games import get_game  # noqa: E402
from tr.inspect import _ids, head_contributions  # noqa: E402
from tr.policy import default_config, load_policy  # noqa: E402


def move_offset(text, fmt):
    m = re.search(r"[Mm]ove\s+", text)
    if m and fmt.startswith("move"):
        return m.end()
    return max(text.find(fmt.split()[0]), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".cache/inspect-move")
    ap.add_argument("--games", nargs="*", default=["puzzle8", "countdown", "hanoi"])
    ap.add_argument("--label", default="optimal")
    ap.add_argument("--per-class", type=int, default=40, help="states per class per game")
    ap.add_argument("--layers", default="4-27")
    a = ap.parse_args()
    lo, hi = a.layers.split("-")
    layers = list(range(int(lo), int(hi) + 1))
    dirs = torch.load(Path(a.dir) / f"dirs-{a.label}.pt")["dirs"]
    pol = load_policy(default_config(thinking=False, max_new_tokens=8))
    proj = {1: [], 0: []}
    for g in a.games:
        p = Path(a.dir) / f"{g}.pt"
        if not p.exists():
            continue
        d = torch.load(p)
        li = d["label_names"].index(a.label)
        game, ver = get_game(g)
        ep = GameEpisode(game, ver, pol)
        rows = [(m, int(l[li])) for m, l in zip(d["meta"], d["labels"]) if m["action"] is not None]
        for cls in (1, 0):
            for m, _ in [r for r in rows if r[1] == cls][: a.per_class]:
                s = game.decode(m["state"])
                msgs = [Message("system", ep.system()), Message("user", ep.solo_prompt(s))]
                ids = _ids(pol, msgs, enable_thinking=False)
                k = move_offset(m["text"], m["action"])
                prefix = pol.tokenizer(m["text"][:k], return_tensors="pt", add_special_tokens=False).input_ids.to(ids.device)
                ids = torch.cat([ids, prefix], dim=1)
                per_layer = []
                for L in layers:
                    hc = head_contributions(pol, ids, L)[-1]  # (n_heads, d)
                    u = dirs[L].float()
                    u = u / u.norm()
                    per_layer.append(hc @ u)
                proj[cls].append(torch.stack(per_layer))  # (layers, heads)
    P1, P0 = torch.stack(proj[1]), torch.stack(proj[0])
    std = torch.cat([P1, P0]).std(0) + 1e-6
    score = (P1.mean(0) - P0.mean(0)) / std  # (layers, heads)
    flat = [(float(score[i, h]), layers[i], h) for i in range(len(layers)) for h in range(score.shape[1])]
    flat.sort()
    print(f"label={a.label} n_opt={len(P1)} n_wrong={len(P0)} layers={a.layers}")
    print("most pro-optimal heads (layer,head,score): " + " ".join(f"L{l}H{h}:{s:+.2f}" for s, l, h in flat[::-1][:12]))
    print("most pro-wrong heads   (layer,head,score): " + " ".join(f"L{l}H{h}:{s:+.2f}" for s, l, h in flat[:12]))
    torch.save({"layers": layers, "score": score, "P1": P1, "P0": P0}, Path(a.dir) / f"heads-{a.label}.pt")


if __name__ == "__main__":
    main()
