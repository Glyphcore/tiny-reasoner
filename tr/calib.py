"""Calibrated routing edit: a truth readout that fires on the judge readout position and stays silent elsewhere.

The plain rank-one route (tr.analyze, d-route-*) raised judge accuracy but destroyed perplexity: its readout w was only
fitted on judge readout positions, so on ordinary text w.a is unconstrained and writes junk into the logits.
Fix, closed form (weighted ridge least squares, ROME-style use of other keys):
    min_w,c  sum_judge (w.a + c - y)^2  +  mu * sum_generic (w.a + c)^2  +  lam |w|^2
y = +1 legal / -1 illegal at the readout position; generic = every position of unrelated text, chats, game turns, tool
prompts and the non-final positions of judge prompts (target 0 = write nothing).

python -m tr.calib capture | fit
"""
from __future__ import annotations

import json
import sys

import numpy as np
import torch

from tr import probes as P
from tr.analyze import ROOT, save_edit, save_vec
from tr.evalh import probe_states, system_prompt
from tr.games import GAMES
from tr.inspect import ACTS, load
from tr.policy import MODELS, HFPolicy, snapshot

LAYERS = [19, 21, 23, 25]


def texts(policy):
    out = [("prose", p.strip()) for p in (ROOT / "data/calib.txt").read_text().split("\n\n") if p.strip()]
    out += [("chat", policy.render(m)) for m in json.loads((ROOT / "data/calib_chats.json").read_text())]
    for g in GAMES.values():
        for s in probe_states(g, "dev", 4):
            m = g.optimal_moves(s)[0] if g.optimal_moves(s) else g.legal_moves(s)[0]
            msgs = [{"role": "system", "content": system_prompt(g, "plain")},
                    {"role": "user", "content": f"State: {g.render(s)}\nYour move?"},
                    {"role": "assistant", "content": f"THINK: I will make progress toward the goal.\nMOVE: {m.text}"},
                    {"role": "user", "content": f"OK. State: {g.render(g.apply(s, m))}\nYour move?"}]
            out.append(("game", policy.render(msgs)))
    out += [("call", policy.render(p.messages)) for p in P.call_probes("dev", 2)]
    out += [("judge_prefix", policy.render(p.messages)) for p in P.judge_probes("fit", 100)[::8]]
    return out


@torch.no_grad()
def capture(policy):
    grab = {}
    hooks = [policy.model.model.layers[L].mlp.down_proj.register_forward_pre_hook(
        lambda m, inp, L=L: grab.__setitem__(L, inp[0][0].float().cpu().numpy())) for L in LAYERS]
    rng = np.random.default_rng(0)
    rows = {L: [] for L in LAYERS}
    kind, text_id = [], []
    try:
        for i, (k, t) in enumerate(texts(policy)):
            ids = policy.tok(t, return_tensors="pt").input_ids.to(policy.device)
            policy.model(input_ids=ids)
            n = ids.shape[1]
            keep = np.arange(n)
            if k == "judge_prefix":  # drop the readout position; subsample, always keep the 8 positions before it
                keep = np.unique(np.concatenate([rng.choice(n - 1, 40, replace=False), np.arange(n - 9, n - 1)]))
            for L in LAYERS:
                rows[L].append(grab[L][keep].astype(np.float16))
            kind += [k] * len(keep)
            text_id += [i] * len(keep)
    finally:
        for h in hooks:
            h.remove()
    np.savez(ACTS / "calib.npz", kind=np.array(kind), text_id=np.array(text_id),
             **{f"L{L}": np.concatenate(rows[L]) for L in LAYERS})
    print(f"captured calib rows={len(kind)}")


def wridge(Xj, y, Xg, mu, lam_rel):
    """Weighted ridge with an unpenalised intercept. Returns w, c."""
    X = np.concatenate([Xj, Xg])
    t = np.concatenate([y, np.zeros(len(Xg))])
    wt = np.concatenate([np.ones(len(Xj)), np.full(len(Xg), mu)])
    m = (wt[:, None] * X).sum(0) / wt.sum()
    Xc, tm = X - m, (wt * t).sum() / wt.sum()
    G = (Xc * wt[:, None]).T @ Xc
    w = np.linalg.solve(G + lam_rel * np.trace(G) / len(G) * np.eye(len(G)), (Xc * wt[:, None]).T @ (t - tm))
    return w, float(tm - m @ w)


def fit():
    from safetensors import safe_open
    from transformers import AutoTokenizer
    path, _ = snapshot(*MODELS["qwen3-0.6b"][:2])
    tok = AutoTokenizer.from_pretrained(path)
    with safe_open(str(path / "model.safetensors"), "pt") as f:
        E = f.get_tensor("model.embed_tokens.weight").float().numpy().astype(np.float64)
    u = E[tok("legal").input_ids[0]] - E[tok("illegal").input_ids[0]]
    un = u / np.linalg.norm(u)
    fit_, dev, cal = load("fit"), load("dev"), dict(np.load(ACTS / "calib.npz"))
    yf, yd = np.where(fit_["truth"], 1.0, -1.0), dev["truth"]
    val = cal["text_id"] % 5 == 0  # hold out every fifth calibration text
    rows = []
    for L in LAYERS:
        Xj, Xd = fit_["mlp"][:, L].astype(np.float64), dev["mlp"][:, L].astype(np.float64)
        Xg, Xv = cal[f"L{L}"][~val].astype(np.float64), cal[f"L{L}"][val].astype(np.float64)
        for rho in (1.0, 10.0, 100.0):
            for lam in (1e-3, 1e-2, 1e-1):
                w, c = wridge(Xj, yf, Xg, rho * len(Xj) / len(Xg), lam)
                out_d, out_v = Xd @ w + c, Xv @ w + c
                rows.append({"layer": L, "rho": rho, "lam": lam, "dev_acc": float(((out_d > 0) == yd).mean()),
                             "judge_amp": float(np.abs(out_d).mean()), "generic_rms": float(np.sqrt((out_v ** 2).mean())),
                             "generic_max": float(np.abs(out_v).max()),
                             "plain_generic_rms": None})
    # the uncalibrated readout on the same held-out generic positions, for the record
    from tr.analyze import ridge
    rep = json.loads((ROOT / "runs/inspect-judge/report.json").read_text())
    r0 = rep["route_candidates"][0]
    w0, b0 = ridge(fit_["mlp"][:, r0["layer"]].astype(np.float64), fit_["truth"], r0["ridge_lam"])
    v0 = cal[f"L{r0['layer']}"][val].astype(np.float64) @ w0 - b0
    plain = {"layer": r0["layer"], "generic_rms": float(np.sqrt((v0 ** 2).mean())), "generic_max": float(np.abs(v0).max())}
    # rule (fixed before any heldout run): leak ratio = generic rms / judge amplitude must stay under a cap, then best dev
    # accuracy, then larger amplitude. Two caps: tight 0.35, loose 0.7. No setting got under 0.15.
    chosen = {}
    for tag, cap in (("tight", 0.35), ("loose", 0.7)):
        best = max([r for r in rows if r["generic_rms"] <= cap * r["judge_amp"]], key=lambda r: (r["dev_acc"], r["judge_amp"]))
        chosen[tag] = best
        L = best["layer"]
        w, c = wridge(fit_["mlp"][:, L].astype(np.float64), yf, cal[f"L{L}"][~val].astype(np.float64),
                      best["rho"] * len(yf) / int((~val).sum()), best["lam"])
        rms = float(np.sqrt((fit_["resid"][:, L + 1].astype(np.float64) ** 2).sum(1).mean()))
        amp = best["judge_amp"]
        A, B = save_vec(f"routecal_{tag}_A", (w / amp)[None, :]), save_vec(f"routecal_{tag}_B", (un * rms)[:, None])
        bias = save_vec(f"routecal_{tag}_bias", c / amp * un * rms)
        for s in (0.2, 0.4, 0.8):
            save_edit(f"d2-routecal-{tag}-L{L}-s{s}", "lowrank",
                      f"rank-one LoRA on layer {L} mlp.down_proj: A = calibrated truth readout on MLP hidden (weighted ridge, target "
                      f"0 on generic positions; dev acc {best['dev_acc']:.2f}, leak ratio {best['generic_rms'] / amp:.2f}), "
                      f"B = unembed(legal) - unembed(illegal); unit readout writes {s} x residual norm",
                      [{"op": "lowrank_add", "tensor": f"model.layers.{L}.mlp.down_proj.weight", "A": A, "B": B, "scale": s},
                       {"op": "bias_add", "module": f"model.layers.{L}.mlp.down_proj", "vec": bias, "scale": s}])
        print(f"{tag}: L{L} rho={best['rho']} lam={best['lam']} dev_acc={best['dev_acc']:.3f} amp={amp:.2f} "
              f"leak={best['generic_rms'] / amp:.2f} max={best['generic_max'] / amp:.1f}")
    (ROOT / "runs/inspect-judge/calib.json").write_text(json.dumps(
        {"n_generic_fit": int((~val).sum()), "n_generic_val": int(val.sum()), "plain_route": plain, "chosen": chosen, "grid": rows}, indent=1))
    print(f"plain L{plain['layer']} generic_rms={plain['generic_rms']:.2f} max={plain['generic_max']:.1f}")
    for Lx in LAYERS:
        print(Lx, " ".join(f"{r['rho']:g}/{r['lam']:g}:{r['dev_acc']:.2f},{r['generic_rms'] / r['judge_amp']:.2f}" for r in rows if r["layer"] == Lx))


if __name__ == "__main__":
    if sys.argv[1] == "capture":
        ACTS.mkdir(parents=True, exist_ok=True)
        capture(HFPolicy())
    else:
        fit()
