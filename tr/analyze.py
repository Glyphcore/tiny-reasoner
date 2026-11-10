"""Read the small model: closed-form probes on cached activations, head attribution, and the edits inferred from them.

python -m tr.analyze            -> runs/inspect-judge/report.json (small, committed), edits/*.json, edits/vec/*.npy
All fits are closed form (mean difference, ridge least squares). No gradients.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

from tr.inspect import load
from tr.policy import MODELS, snapshot

ROOT = Path(__file__).resolve().parent.parent
EDITS, VEC = ROOT / "edits", ROOT / "edits" / "vec"
TOK = {"legal": 6428, "illegal": 0}  # filled from the tokenizer in main()
NH, HD = 16, 128


def said_legal(a):
    lg = a["logits"]
    return np.maximum(lg[:, 0], lg[:, 2]) > np.maximum(lg[:, 1], lg[:, 3])


def ld(a):
    return a["logits"][:, 0] - a["logits"][:, 1]


def meandiff(X, y):
    mu1, mu0 = X[y].mean(0), X[~y].mean(0)
    w = mu1 - mu0
    return w, float((mu1 + mu0) @ w / 2)


def ridge(X, y, lam_rel):
    mu = X.mean(0)
    Xc, t = X - mu, np.where(y, 1.0, -1.0)
    K = Xc @ Xc.T
    if np.trace(K) < 1e-9:  # identical inputs (e.g. the embedding of the same last token)
        return np.zeros(X.shape[1]), 0.0
    alpha = np.linalg.solve(K + lam_rel * np.trace(K) / len(K) * np.eye(len(K)), t - t.mean())
    w = Xc.T @ alpha
    return w, float(mu @ w - t.mean())


def acc(X, y, w, b):
    return float((((X @ w) - b > 0) == y).mean())


def ridge_cv(X, y, lams=(1e-3, 1e-2, 1e-1, 1.0), k=5):
    idx = np.random.default_rng(0).permutation(len(y))
    best = (-1, None)
    for lam in lams:
        a = []
        for f in range(k):
            te = idx[f::k]
            tr = np.setdiff1d(idx, te)
            w, b = ridge(X[tr], y[tr], lam)
            a.append(acc(X[te], y[te], w, b))
        if np.mean(a) > best[0]:
            best = (float(np.mean(a)), lam)
    return best


def save_vec(name, v):
    VEC.mkdir(parents=True, exist_ok=True)
    p = VEC / f"{name}.npy"
    np.save(p, np.asarray(v, dtype=np.float32))
    return str(p.relative_to(ROOT))


def save_edit(eid, family, desc, ops):
    EDITS.mkdir(exist_ok=True)
    (EDITS / f"{eid}.json").write_text(json.dumps({"id": eid, "family": family, "description": desc, "ops": ops}, indent=1))


def main():
    from transformers import AutoTokenizer
    path, _ = snapshot(*MODELS["qwen3-0.6b"][:2])
    tok = AutoTokenizer.from_pretrained(path)
    t_legal, t_illegal = tok("legal").input_ids[0], tok("illegal").input_ids[0]
    fit, dev = load("fit"), load("dev")
    yf, yd = fit["truth"], dev["truth"]
    rep = {"n_fit": len(yf), "n_dev": len(yd), "positions": {"resid/mlp/head": "last prompt token (answer readout)",
                                                         "resid_move": "last token of the proposed move"}}

    # A. what the output says
    sf, sd = said_legal(fit), said_legal(dev)
    thr = np.sort(ld(fit))
    best_thr = max(thr, key=lambda c: ((ld(fit) > c) == yf).mean())
    rep["output"] = {"fit_acc": float((sf == yf).mean()), "dev_acc": float((sd == yd).mean()),
                     "dev_says_legal_rate": float(sd.mean()),
                     "dev_acc_logit_diff_rethresholded": float(((ld(dev) > best_thr) == yd).mean()),
                     "logit_diff_threshold": float(best_thr),
                     "dev_acc_per_game": {g: float((sd == yd)[dev["game"] == g].mean()) for g in np.unique(dev["game"])}}

    # B. what the activations know: probes for TRUE legality per layer
    layers = fit["mlp"].shape[1]
    rep["probe"] = []
    for src, n in (("resid", layers + 1), ("resid_move", layers + 1), ("mlp", layers)):
        for l in range(n):
            Xf, Xd = fit[src][:, l].astype(np.float64), dev[src][:, l].astype(np.float64)
            w, b = meandiff(Xf, yf)
            cv, lam = ridge_cv(Xf, yf)
            wr, br = ridge(Xf, yf, lam)
            rep["probe"].append({"src": src, "layer": l, "meandiff_dev": acc(Xd, yd, w, b), "ridge_cv_fit": cv,
                                 "ridge_lam": lam, "ridge_dev": acc(Xd, yd, wr, br)})

    # C. head attribution: direct logit attribution to (legal - illegal), signed by truth
    with safe_open(str(path / "model.safetensors"), "pt") as f:
        E = f.get_tensor("model.embed_tokens.weight").float().numpy().astype(np.float64)
        u = E[t_legal] - E[t_illegal]
        heads = []
        sign = np.where(yf, 1.0, -1.0)
        for l in range(layers):
            WO = f.get_tensor(f"model.layers.{l}.self_attn.o_proj.weight").float().numpy().astype(np.float64)
            uw = u @ WO  # [2048]
            z = fit["head"][:, l].astype(np.float64)
            for h in range(NH):
                dla = z[:, h * HD:(h + 1) * HD] @ uw[h * HD:(h + 1) * HD]
                heads.append({"layer": l, "head": h, "helpful": float((sign * dla).mean()), "bias_to_legal": float(dla.mean())})
    rep["heads_most_harmful"] = sorted(heads, key=lambda r: r["helpful"])[:8]
    rep["heads_most_helpful"] = sorted(heads, key=lambda r: -r["helpful"])[:8]
    rep["heads_most_legal_bias"] = sorted(heads, key=lambda r: -r["bias_to_legal"])[:8]
    un = u / np.linalg.norm(u)

    # D. edits inferred from the inspection ------------------------------------------------------------------------
    for old in EDITS.glob("*.json"):
        old.unlink()
    right = sf == yf
    p_true = 1 / (1 + np.exp(-np.where(yf, 1, -1) * ld(fit)))
    wgt = 2 * p_true - 1  # verifier agreement in [-1, 1]
    rep["directions"] = {}
    rep["cells_fit"] = {"says_legal_is_legal": int((sf & yf).sum()), "says_legal_is_illegal": int((sf & ~yf).sum()),
                        "says_illegal_is_legal": int((~sf & yf).sum()), "says_illegal_is_illegal": int((~sf & ~yf).sum())}
    for L in (12, 20):  # bias on layer L's mlp.down_proj lands in resid[L + 1]
        X = fit["resid"][:, L + 1].astype(np.float64)
        mod = f"model.layers.{L}.mlp.down_proj"
        d2 = X[right].mean(0) - X[~right].mean(0)  # (ii) self-report: accurate minus inaccurate
        d3 = (wgt[:, None] * (X - X.mean(0))).sum(0) / np.abs(wgt).sum()  # (iii) weighted by verifier agreement
        # (ii, class-balanced): accurate-minus-inaccurate inside each truth class, averaged; removes the truth confound
        d2b = ((X[sf & yf].mean(0) - X[~sf & yf].mean(0)) + (X[~sf & ~yf].mean(0) - X[sf & ~yf].mean(0))) / 2
        dt, _ = meandiff(X, yf)
        ds = X[sf].mean(0) - X[~sf].mean(0)  # what the model says
        rms = float(np.sqrt((X ** 2).sum(1).mean()))
        cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))  # noqa: E731
        rep["directions"][L] = {"resid_norm": rms, "selfreport_norm": float(np.linalg.norm(d2)),
                                "balanced_norm": float(np.linalg.norm(d2b)),
                                "cos_selfreport_weighted": cos(d2, d3), "cos_selfreport_truth": cos(d2, dt),
                                "cos_selfreport_says": cos(d2, ds), "cos_balanced_truth": cos(d2b, dt),
                                "cos_balanced_says": cos(d2b, ds), "cos_selfreport_unembed": cos(d2, un)}
        vecs = {"a2-selfreport": (d2, "self-report accurate-minus-inaccurate direction"),
                "a2b-balanced": (d2b, "class-balanced self-report direction (accurate minus inaccurate within each truth class)"),
                "a3-weighted": (d3, "verifier-agreement-weighted contrast")}
        for name, (d, what) in vecs.items():
            v = save_vec(f"{name}_L{L}", d / np.linalg.norm(d) * rms)
            for sc in ((0.1, 0.3, -0.1, -0.3) if name != "a3-weighted" else (0.1, 0.3)):
                save_edit(f"{name}-L{L}-s{sc}", "steer_bias", f"{what} baked as a bias on layer {L} mlp.down_proj, "
                          f"{sc} x residual norm", [{"op": "bias_add", "module": mod, "vec": v, "scale": sc}])
            if L == 12 and name != "a3-weighted":
                writers = [r"model\.layers\.\d+\.self_attn\.o_proj\.weight", r"model\.layers\.\d+\.mlp\.down_proj\.weight"]
                save_edit(f"b-ortho-{name[3:]}-L{L}", "ortho", f"{what} (layer {L}) orthogonalised out of every o_proj and "
                          "down_proj (abliteration recipe)", [{"op": "ortho", "tensors": writers, "vec": v, "strength": 1.0}])
    # (a1) the output has a constant yes-bias (re-thresholding the logit difference alone gains accuracy): shift it
    rms = float(np.sqrt((fit["resid"][:, layers].astype(np.float64) ** 2).sum(1).mean()))
    v = save_vec("debias_L27", -un * rms)
    for sc in (0.02, 0.05, 0.1):
        save_edit(f"a1-debias-L{layers - 1}-s{sc}", "steer_bias", "constant shift along unembed(illegal) - unembed(legal) baked as "
                  f"a bias on the last mlp.down_proj, {sc} x residual norm",
                  [{"op": "bias_add", "module": f"model.layers.{layers - 1}.mlp.down_proj", "vec": v, "scale": sc}])

    # (i) route the truth direction into the answer logits: rank-one LoRA on a late down_proj, readout w on MLP hidden
    mlp_rows = [r for r in rep["probe"] if r["src"] == "mlp" and r["layer"] >= 14]
    rep["route_candidates"] = sorted(mlp_rows, key=lambda r: -r["ridge_dev"])[:5]
    for r in rep["route_candidates"][:1]:
        L = r["layer"]
        Xf = fit["mlp"][:, L].astype(np.float64)
        w, b = ridge(Xf, yf, r["ridge_lam"])
        rms = float(np.sqrt((fit["resid"][:, L + 1].astype(np.float64) ** 2).sum(1).mean()))
        A, B = save_vec(f"route_L{L}_A", w[None, :]), save_vec(f"route_L{L}_B", (un * rms)[:, None])
        bias = save_vec(f"route_L{L}_bias", -b * un * rms)
        for s in (0.05, 0.1, 0.2, 0.4):
            save_edit(f"d-route-L{L}-s{s}", "lowrank", f"rank-one LoRA on layer {L} mlp.down_proj: A = ridge truth probe on MLP "
                      f"hidden (dev acc {r['ridge_dev']:.2f}), B = unembed(legal) - unembed(illegal); probe output +-1 writes "
                      f"{s} x residual norm; threshold as bias",
                      [{"op": "lowrank_add", "tensor": f"model.layers.{L}.mlp.down_proj.weight", "A": A, "B": B, "scale": s},
                       {"op": "bias_add", "module": f"model.layers.{L}.mlp.down_proj", "vec": bias, "scale": s}])

    # (c) heads: scale down the head that writes `legal` regardless of truth; ablate harmful; boost helpful
    def cols(r, s):
        return {"op": "scale_cols", "tensor": f"model.layers.{r['layer']}.self_attn.o_proj.weight",
                "start": r["head"] * HD, "end": (r["head"] + 1) * HD, "scale": s}
    top = rep["heads_most_legal_bias"][0]
    for sc in (0.75, 0.5, 0.0):
        save_edit(f"c-yeshead-L{top['layer']}H{top['head']}-x{sc}", "scale", f"scale head L{top['layer']}H{top['head']} (largest "
                  f"truth-independent push toward `legal`) by {sc}", [cols(top, sc)])
    save_edit("c-ablate-harmful4", "scale", "zero the 4 heads with the most negative truth-signed attribution",
              [cols(r, 0.0) for r in rep["heads_most_harmful"][:4]])
    save_edit("c-boost-helpful4-x1.5", "scale", "scale the 4 heads with the most positive truth-signed attribution by 1.5",
              [cols(r, 1.5) for r in rep["heads_most_helpful"][:4]])

    out = ROOT / "runs" / "inspect-judge"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    o = rep["output"]
    best = lambda src, key: max((r for r in rep["probe"] if r["src"] == src), key=lambda r: r[key])  # noqa: E731
    br, bm = best("resid", "ridge_dev"), best("mlp", "ridge_dev")
    print(f"output dev_acc={o['dev_acc']:.3f} says_legal={o['dev_says_legal_rate']:.3f} rethresh={o['dev_acc_logit_diff_rethresholded']:.3f} | "
          f"probe resid L{br['layer']} ridge={br['ridge_dev']:.3f} md={br['meandiff_dev']:.3f} | mlp L{bm['layer']} ridge={bm['ridge_dev']:.3f}")


if __name__ == "__main__":
    main()
