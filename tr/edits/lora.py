"""Hand-built low-rank deltas in LoRA form. A and B come from analysis (directions, SVD of a computed delta), never gradients.

merge folds alpha * B @ A into each target weight through a TensorEdit (kind lora_merge), so revert is exact.
"""
from __future__ import annotations

import json
import os
from typing import Sequence

from spec.interfaces import Adapter, LoraTarget, Policy, WeightEdit

from tr.edits.weightedit import TensorEdit, make_edit


class LoraAdapter(Adapter):
    def __init__(self, id: str, targets: Sequence[LoraTarget]):
        assert targets, "adapter needs at least one target"
        ranks = {t.A.shape[0] for t in targets}
        for t in targets:
            assert t.B.shape[1] == t.A.shape[0], f"{t.tensor}: B is (out, r), A is (r, in)"
        self.id = id
        self.rank = max(ranks)
        self.targets = list(targets)

    def merge(self, policy: Policy) -> WeightEdit:
        import torch
        fns = {}
        for t in self.targets:
            A, B, alpha = t.A.detach().float(), t.B.detach().float(), float(t.alpha)

            def fn(w, A=A, B=B, alpha=alpha):
                return w + alpha * (B.to(w.device) @ A.to(w.device))
            fns[t.tensor] = (fn, {"op": "lora_merge", "adapter": self.id, "rank": int(A.shape[0]), "alpha": alpha})
        return make_edit(f"{self.id}-merged", "lora_merge", policy, fns)

    def save(self, path: str) -> None:
        from safetensors.torch import save_file
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {"id": self.id, "rank": self.rank,
                "targets": [{"tensor": t.tensor, "alpha": float(t.alpha)} for t in self.targets]}
        with open(path + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        tensors = {}
        for i, t in enumerate(self.targets):
            tensors[f"{i}/A"] = t.A.detach().cpu().float().contiguous()
            tensors[f"{i}/B"] = t.B.detach().cpu().float().contiguous()
        save_file(tensors, path + ".safetensors")

    def load(self, path: str) -> "LoraAdapter":
        return load_adapter(path)


def load_adapter(path: str) -> LoraAdapter:
    from safetensors.torch import load_file
    with open(path + ".json") as f:
        meta = json.load(f)
    tensors = load_file(path + ".safetensors")
    targets = [LoraTarget(tensor=m["tensor"], A=tensors[f"{i}/A"], B=tensors[f"{i}/B"], alpha=m["alpha"])
               for i, m in enumerate(meta["targets"])]
    return LoraAdapter(meta["id"], targets)
