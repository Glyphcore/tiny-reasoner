#!/usr/bin/env python3
"""Collect residual activations at the answer position for game states, labelled by what the model's greedy move does.

Held-in seeds (default 100-139) only; eval seeds 0-9 stay untouched. For every state on the optimal trajectory of each
instance (plus one random legal detour per state) we build the exact solo prompt, generate the model's move in
non-thinking mode, label it (parsed / legal / optimal) and store the residual stream at the last prompt token for every
layer. Output: .cache/inspect/<game>.pt with X (n, L+1, d) fp16, labels and metadata. Prints one summary line per game.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from spec.interfaces import Message  # noqa: E402
from tr.episode import GameEpisode  # noqa: E402
from tr.games import get_game  # noqa: E402
from tr.inspect import _ids, residuals  # noqa: E402
from tr.policy import default_config, load_policy  # noqa: E402


def optimal_actions(game, state):
    n = game.optimal_steps(state)
    return {a for a in game.legal_moves(state) if game.optimal_steps(game.apply(state, a)) == n - 1}


def states_for(game, level, seed, rng):
    import importlib
    mod = importlib.import_module(game.__class__.__module__)
    s = game.new(level, seed)
    out = []
    for a in mod.solve(s):
        out.append(s)
        legal = list(game.legal_moves(s))
        if len(legal) > 1:
            det = rng.choice([m for m in legal if m != a])
            d = game.apply(s, det)
            try:
                if game.optimal_steps(d) > 0:
                    out.append(d)
            except ValueError:
                pass  # unsolvable detour (countdown)
        s = game.apply(s, a)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", nargs="*", default=["hanoi:2,3,4", "puzzle8:1,2,3", "countdown:1,2,3"])
    ap.add_argument("--seeds", default="100-139")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default=".cache/inspect")
    a = ap.parse_args()
    lo, hi = a.seeds.split("-")
    seeds = range(int(lo), int(hi) + 1)
    pol = load_policy(default_config(thinking=False, max_new_tokens=a.max_new_tokens))
    n_layers = len(pol.model.model.layers)
    layers = list(range(-1, n_layers))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    for spec in a.games:
        name, _, lv = spec.partition(":")
        levels = [int(x) for x in lv.split(",")]
        game, ver = get_game(name)
        ep = GameEpisode(game, ver, pol)
        rng = random.Random(0)
        seen, X, lab, meta = set(), [], [], []
        t0 = time.time()
        for level in levels:
            for seed in seeds:
                for s in states_for(game, level, seed, rng):
                    key = (level, game.encode(s).__repr__())
                    if key in seen:
                        continue
                    seen.add(key)
                    msgs = [Message("system", ep.system()), Message("user", ep.solo_prompt(s))]
                    g = pol.generate(msgs)
                    act = game.parse(g.text)
                    legal = act is not None and ver.legal(s, act)
                    opt = legal and act in optimal_actions(game, s)
                    ids = _ids(pol, msgs, enable_thinking=False)
                    r = residuals(pol, ids, layers)
                    X.append(torch.stack([r[l][-1] for l in layers]).half())
                    lab.append((int(act is not None), int(legal), int(opt)))
                    meta.append({"level": level, "seed": seed, "state": game.encode(s), "text": g.text[:200],
                                 "action": None if act is None else game.format(act), "optimal_steps": game.optimal_steps(s)})
        X = torch.stack(X)
        L = torch.tensor(lab)
        torch.save({"X": X, "labels": L, "label_names": ["parsed", "legal", "optimal"], "meta": meta, "layers": layers,
                    "model": pol.config.model, "revision": pol.config.revision}, f"{a.out}/{name}.pt")
        m = L.float().mean(0)
        print(f"{name}: n={len(L)} parsed={m[0]:.2f} legal={m[1]:.2f} optimal={m[2]:.2f} {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
