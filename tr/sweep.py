"""Cheap dev sweep over edits: first-token judge accuracy on the dev probes (no generation) + perplexity, then revert.

python -m tr.sweep <glob of edits/*.json> ...   -> appends runs/sweep/sweep.jsonl, prints one line per edit
Selection rule (written before looking at heldout): the largest scale whose ppl <= 1.10 x base and dev acc is best.
"""
from __future__ import annotations

import glob
import json
import sys
import time

import numpy as np

from tr import probes as P
from tr.edits import apply_edit, model_hash
from tr.evalh import ROOT, perplexity
from tr.inspect import SIZES, answer_tokens
from tr.policy import HFPolicy
from tr.run import load_edit


def quick(policy, probes):
    toks = answer_tokens(policy)
    says, truth, games = [], [], []
    for p in probes:
        lg = policy.forward_text(policy.render(p.messages))
        says.append(max(lg[toks["legal"]], lg[toks["Legal"]]) > max(lg[toks["illegal"]], lg[toks["Illegal"]]))
        truth.append(p.meta["truth"])
        games.append(p.meta["game"])
        top = int(lg.argmax())
    says, truth, games = np.array(says), np.array(truth), np.array(games)
    r = {"dev_acc": float((says == truth).mean()), "says_legal": float(says.mean()), "last_top": policy.tok.decode([top])}
    r.update({f"acc_{g}": float((says == truth)[games == g].mean()) for g in np.unique(games)})
    return r


def main():
    policy = HFPolicy()
    probes = P.judge_probes("dev", SIZES["dev"])
    base = model_hash(policy.model)
    out = ROOT / "runs" / "sweep"
    out.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["edit"] for l in open(out / "sweep.jsonl")} if (out / "sweep.jsonl").exists() else set()
    paths = ([] if "base" in done else ["none"]) + [p for a in sys.argv[1:] for p in sorted(glob.glob(a))
                                                     if json.loads(open(p).read())["id"] not in done]
    paths = list(dict.fromkeys(paths))
    with open(out / "sweep.jsonl", "a") as f:
        for path in paths:
            t0 = time.time()
            edit = None if path == "none" else load_edit(path)
            applied = apply_edit(policy.model, edit) if edit else None
            try:
                r = quick(policy, probes)
                r["ppl"], _ = perplexity(policy)
            finally:
                if applied:
                    applied.revert()
            r.update(edit=edit.id if edit else "base", secs=round(time.time() - t0, 1))
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{r['edit']} acc={r['dev_acc']:.3f} says_legal={r['says_legal']:.2f} ppl={r['ppl']:.2f} "
                  f"h={r['acc_hanoi']:.2f} c={r['acc_countdown']:.2f} g={r['acc_grid']:.2f} top={r['last_top']!r}", flush=True)
    assert model_hash(policy.model) == base
    print("revert_exact=True")


if __name__ == "__main__":
    main()
