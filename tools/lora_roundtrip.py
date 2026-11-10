"""Hand-built LoRA on the real model: unmerged side path vs merged weights, exact unmerge, and why revert uses a backup.

python tools/lora_roundtrip.py edits/<lowrank edit>.json  -> runs/lora-roundtrip/{config.json,metrics.jsonl}, one line
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from spec.interfaces import Adapter, WeightEdit  # noqa: E402
from tr import probes as P  # noqa: E402
from tr.edits import apply_edit, attach, model_hash, tensor_hash  # noqa: E402
from tr.evalh import ROOT  # noqa: E402
from tr.policy import HFPolicy  # noqa: E402
from tr.run import load_edit  # noqa: E402


def main():
    edit = load_edit(sys.argv[1])
    low = [o for o in edit.ops if o["op"] == "lowrank_add"][0]
    bias_ops = [o for o in edit.ops if o["op"] == "bias_add"]
    policy = HFPolicy()
    base = model_hash(policy.model)
    A, B = np.load(ROOT / low["A"]), np.load(ROOT / low["B"])
    ad = Adapter(edit.id, A.shape[0], low["scale"] * A.shape[0], {low["tensor"]: (A, B)})
    texts = [policy.render(p.messages) for p in P.judge_probes("dev", 50)[::5]]

    def logits():
        return torch.stack([policy.forward_text(t) for t in texts])
    y0 = logits()
    bias_edit = WeightEdit("bias-only", "steer_bias", "threshold term of the adapter", bias_ops)
    applied_b = apply_edit(policy.model, bias_edit)
    detach = attach(policy.model, ad)
    y_side = logits()  # unmerged: W untouched, side path B A x
    detach()
    applied_b.revert()
    assert model_hash(policy.model) == base
    applied = apply_edit(policy.model, edit)
    y_merged = logits()
    w = dict(policy.model.named_parameters())[low["tensor"]]
    with torch.no_grad():  # arithmetic unmerge W + sBA - sBA is not bit exact in floating point
        arith = (w.float() - low["scale"] * torch.from_numpy(B) @ torch.from_numpy(A)).to(w.dtype)
    arith_exact = tensor_hash(arith) == edit.before[low["tensor"]]
    applied.revert()
    arith_err = float((arith - dict(policy.model.named_parameters())[low["tensor"]]).abs().max())
    reverted = model_hash(policy.model)
    row = {"edit": edit.id, "n_prompts": len(texts), "rank": int(A.shape[0]),
           "max_abs_logit_change_vs_base": float((y_merged - y0).abs().max()),
           "max_abs_logit_diff_merged_vs_unmerged": float((y_merged - y_side).abs().max()),
           "argmax_agree_merged_vs_unmerged": float((y_merged.argmax(1) == y_side.argmax(1)).float().mean()),
           "arithmetic_unmerge_bit_exact": arith_exact, "arithmetic_unmerge_max_abs_err": arith_err,
           "backup_revert_exact": reverted == base}
    out = ROOT / "runs" / "lora-roundtrip"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"id": "lora-roundtrip", "model": policy.model_id, "revision": policy.revision,
                                                 "edit": sys.argv[1], "device": policy.device, "dtype": str(policy.dtype),
                                                 "base_hash": base, "reverted_hash": reverted, "before": edit.before,
                                                 "after": edit.after}, indent=1))
    (out / "metrics.jsonl").write_text(json.dumps(row) + "\n")
    print(" ".join(f"{k}={v}" for k, v in row.items()))


if __name__ == "__main__":
    main()
