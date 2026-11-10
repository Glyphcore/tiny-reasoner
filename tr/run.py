"""One run = one variant (base, prompt-only, or one WeightEdit) evaluated, then reverted.

python -m tr.run --id base --split heldout [--prompt hint] [--edit edits/x.json] [--parts gpxc]
Writes runs/<id>/config.json, metrics.jsonl, chats/*.md; prints a one-line summary.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

from spec.interfaces import Run, WeightEdit
from tr.edits import apply_edit, model_hash
from tr.evalh import ROOT, evaluate
from tr.policy import HFPolicy


def load_edit(path):
    d = json.loads(Path(path).read_text())
    return WeightEdit(d["id"], d["family"], d["description"], d["ops"])


def run(policy, run_id, split="heldout", prompt="plain", edit=None, parts="jtgpxc", base_hash=None, **kw):
    t0 = time.time()
    outdir = ROOT / "runs" / run_id
    base_hash = base_hash or model_hash(policy.model)
    applied = apply_edit(policy.model, edit) if edit else None
    try:
        ev = evaluate(policy, split, prompt, outdir, parts=parts, **kw)
    finally:
        if applied:
            applied.revert()
    reverted = model_hash(policy.model) if edit else base_hash
    assert reverted == base_hash, "revert did not restore base hashes"
    r = Run(run_id, policy.model_id, policy.revision, policy.license, policy.device, str(policy.dtype), policy.threads,
            prompt, edit, ev, base_hash, reverted, round(time.time() - t0, 1))
    cfg = asdict(r)
    cfg.update(decoding="greedy", parts=parts, training="none (computed edits only)")
    (outdir / "config.json").write_text(json.dumps(cfg, indent=1))
    c = ev.confusion
    line = (f"{run_id} split={split} judge_acc={ev.judge_acc:.3f} hit={ev.judge_hit:.3f} valid={ev.judge_valid:.3f} "
            f"LL={c.get('says_legal_is_legal')} LI={c.get('says_legal_is_illegal')} IL={c.get('says_illegal_is_legal')} "
            f"II={c.get('says_illegal_is_illegal')} call_hit={ev.call_hit:.3f} call_valid={ev.call_valid:.3f} solve={ev.solve:.3f} legal={ev.legal:.3f} opt={ev.optimal:.3f} recover={ev.recover:.3f} "
            f"steps={ev.steps:.2f} trace={ev.trace_len:.1f} probe_legal={ev.probe_legal:.3f} probe_opt={ev.probe_opt:.3f} "
            f"ppl={ev.ppl:.2f} chats={ev.chats_ok} revert_exact={reverted == base_hash} secs={r.seconds}")
    (outdir / "summary.txt").write_text(line + "\n")
    return line, base_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", action="append", required=True)
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--prompt", default="plain")
    ap.add_argument("--edit", action="append", default=None, help="edit json per --id ('none' for no edit)")
    ap.add_argument("--parts", default="jtgpxc")
    ap.add_argument("--probe-n", type=int, default=20)
    ap.add_argument("--judge-n", type=int, default=50)
    ap.add_argument("--call-n", type=int, default=10)
    a = ap.parse_args()
    policy = HFPolicy()
    edits = a.edit or ["none"] * len(a.id)
    base_hash = None
    for rid, e in zip(a.id, edits):
        line, base_hash = run(policy, rid, a.split, a.prompt, None if e == "none" else load_edit(e), a.parts,
                              base_hash=base_hash, probe_n=a.probe_n, judge_n=a.judge_n, call_n=a.call_n)
        print(line, flush=True)


if __name__ == "__main__":
    main()
