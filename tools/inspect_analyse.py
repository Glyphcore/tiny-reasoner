#!/usr/bin/env python3
"""Analyse collected activations: per layer, how well a diff-of-means direction separates optimal (or legal) from other
moves, on a held split (seeds by parity). Saves directions to .cache/inspect/dirs.pt. Prints a compact table only."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402


def auc(scores, y):
    """ROC AUC by rank statistic."""
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".cache/inspect")
    ap.add_argument("--games", nargs="*", default=["hanoi", "puzzle8", "countdown"])
    ap.add_argument("--label", default="optimal", choices=["parsed", "legal", "optimal"])
    a = ap.parse_args()
    data = {}
    for g in a.games:
        p = Path(a.dir) / f"{g}.pt"
        if p.exists():
            data[g] = torch.load(p)
    if not data:
        sys.exit("no data")
    li = data[next(iter(data))]["label_names"].index(a.label)
    layers = data[next(iter(data))]["layers"]
    dirs = {}
    rows = []
    # pooled, with per-game train/held split by seed parity
    Xtr, ytr, Xhe, yhe = [], [], [], []
    for g, d in data.items():
        X, y = d["X"].float(), d["labels"][:, li]
        seed = torch.tensor([m["seed"] for m in d["meta"]])
        tr, he = seed % 2 == 0, seed % 2 == 1
        Xtr.append(X[tr]); ytr.append(y[tr]); Xhe.append(X[he]); yhe.append(y[he])
        rows.append((g, X, y, tr, he))
    Xtr, ytr, Xhe, yhe = torch.cat(Xtr), torch.cat(ytr), torch.cat(Xhe), torch.cat(yhe)
    print(f"label={a.label} n_train={len(ytr)} pos_rate={ytr.float().mean():.2f} n_held={len(yhe)}")
    hdr = "layer " + " ".join(f"{g[:6]:>6}" for g in data) + "  pooled  |d|/|x|"
    print(hdr)
    for j, l in enumerate(layers):
        dpool = Xtr[ytr == 1, j].mean(0) - Xtr[ytr == 0, j].mean(0)
        dirs[l] = dpool
        cells = []
        for g, X, y, tr, he in rows:
            d = X[tr][y[tr] == 1, j].mean(0) - X[tr][y[tr] == 0, j].mean(0)
            cells.append(auc(X[he, j] @ d, y[he]))
        pooled = auc(Xhe[:, j] @ dpool, yhe)
        rel = float(dpool.norm() / Xtr[:, j].norm(dim=1).mean())
        print(f"{l:>5} " + " ".join(f"{c:6.2f}" for c in cells) + f"  {pooled:6.2f}  {rel:.3f}")
    torch.save({"label": a.label, "layers": layers, "dirs": dirs}, Path(a.dir) / f"dirs-{a.label}.pt")


if __name__ == "__main__":
    main()
