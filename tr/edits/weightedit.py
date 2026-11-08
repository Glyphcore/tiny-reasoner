"""Direct tensor edits with exact revert.

TensorEdit stores, per edited tensor, a CPU copy of the tensor before and after the change. apply copies `after` into the
live parameter, revert copies `before` back, so the before-hashes are restored bit-exactly (no fp16 add/subtract round trip).
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from typing import Callable, Mapping

from spec.interfaces import JSON, Policy, Tensor, TensorChange, WeightEdit


def tensor_hash(t: Tensor) -> str:
    import torch
    x = t.detach()
    if x.dtype != torch.float16:
        x = x.to(torch.float16)
    b = x.contiguous().cpu().numpy().tobytes()
    h = hashlib.sha256(str(tuple(x.shape)).encode())
    h.update(b)
    return h.hexdigest()


class TensorEdit(WeightEdit):
    def __init__(self, id: str, kind: str, before: Mapping[str, Tensor], after: Mapping[str, Tensor],
                 how: Mapping[str, JSON]):
        assert set(before) == set(after) == set(how), "before/after/how must name the same tensors"
        self.id = id
        self.kind = kind
        self._before = {k: v.detach().cpu().clone() for k, v in before.items()}
        self._after = {k: v.detach().cpu().clone() for k, v in after.items()}
        self.changes = [TensorChange(name=k, hash_before=tensor_hash(self._before[k]), hash_after=tensor_hash(self._after[k]),
                                     how=dict(how[k])) for k in sorted(before)]
        self.applied = False

    # -- live model
    def _copy(self, policy: Policy, src: Mapping[str, Tensor]) -> None:
        import torch
        with torch.no_grad():
            for name, t in src.items():
                p = policy.tensor(name)
                p.copy_(t.to(device=p.device, dtype=p.dtype))

    def apply(self, policy: Policy) -> None:
        if not self.verify(policy, "before"):
            raise RuntimeError(f"{self.id}: live tensors do not match hash_before; refusing to apply")
        self._copy(policy, self._after)
        self.applied = True

    def revert(self, policy: Policy) -> None:
        self._copy(policy, self._before)
        self.applied = False
        if not self.verify(policy, "before"):
            raise RuntimeError(f"{self.id}: revert did not restore hash_before")

    def verify(self, policy: Policy, expect: str) -> bool:
        key = "hash_before" if expect == "before" else "hash_after"
        return all(tensor_hash(policy.tensor(c.name)) == getattr(c, key) for c in self.changes)

    # -- disk: <path>.json + <path>.safetensors (before and after tensors in fp16)
    def save(self, path: str) -> None:
        from safetensors.torch import save_file
        import torch
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {"id": self.id, "kind": self.kind, "changes": [asdict(c) for c in self.changes]}
        with open(path + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        tensors = {}
        for k in self._before:
            tensors[f"before/{k}"] = self._before[k].to(torch.float16).contiguous()
            tensors[f"after/{k}"] = self._after[k].to(torch.float16).contiguous()
        save_file(tensors, path + ".safetensors")

    def load(self, path: str) -> "TensorEdit":
        return load_edit(path)


def load_edit(path: str) -> TensorEdit:
    from safetensors.torch import load_file
    with open(path + ".json") as f:
        meta = json.load(f)
    tensors = load_file(path + ".safetensors")
    before = {k[len("before/"):]: v for k, v in tensors.items() if k.startswith("before/")}
    after = {k[len("after/"):]: v for k, v in tensors.items() if k.startswith("after/")}
    how = {c["name"]: c["how"] for c in meta["changes"]}
    edit = TensorEdit(meta["id"], meta["kind"], before, after, how)
    for c, m in zip(edit.changes, meta["changes"]):
        if (c.hash_before, c.hash_after) != (m["hash_before"], m["hash_after"]):
            raise RuntimeError(f"{path}: stored hashes for {c.name} do not match the tensors")
    return edit


def make_edit(id: str, kind: str, policy: Policy, fns: Mapping[str, tuple[Callable[[Tensor], Tensor], JSON]]) -> TensorEdit:
    """Build an edit from per-tensor functions new = fn(old) (computed in fp32, stored in the param dtype)."""
    import torch
    before, after, how = {}, {}, {}
    with torch.no_grad():
        for name, (fn, h) in fns.items():
            p = policy.tensor(name)
            before[name] = p
            after[name] = fn(p.detach().float()).to(p.dtype)
            how[name] = h
    return TensorEdit(id, kind, before, after, how)
