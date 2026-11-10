"""Direct weight edits. Computed, never trained. Every edit records sha256 before/after and reverts from a backup.

Recipe ops (all reproducible from config.json; vectors are small .npy files next to it):
  bias_add     {"module": name, "vec": path, "scale": a}            add a*v to module.bias (created if absent)
  ortho        {"tensors": [names], "vec": path, "strength": s}     W <- W - s * d d^T W   (d unit, output side)
  scale_cols   {"tensor": name, "start": i, "end": j, "scale": s}   W[:, i:j] *= s          (head / neuron)
  lowrank_add  {"tensor": name, "A": path, "B": path, "scale": s}   W += s * B @ A          (LoRA merge)
  task_vector  {"other": model name, "lam": l}                      W += l * (W - W_other) for every float tensor
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import torch

from spec.interfaces import Adapter, WeightEdit


def tensor_hash(t):
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def model_hash(model):
    h = hashlib.sha256()
    for name, p in sorted(model.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def _module(model, name):
    return model.get_submodule(name)


def _vec(path, like):
    return torch.from_numpy(np.load(path)).to(device=like.device, dtype=torch.float32)


def expand(model, patterns):
    names = [n for n, _ in model.named_parameters()]
    return [n for n in names if any(re.fullmatch(p, n) for p in patterns)]


class Applied:
    """Handle for an applied edit: holds the backup and restores it exactly."""

    def __init__(self, model, edit, backup, created):
        self.model, self.edit, self.backup, self.created = model, edit, backup, created

    @torch.no_grad()
    def revert(self):
        sd = dict(self.model.named_parameters())
        for name, old in self.backup.items():
            sd[name].copy_(old)
        for mod_name in self.created:
            _module(self.model, mod_name).bias = None
        now = dict(self.model.named_parameters())
        for name, h in self.edit.before.items():
            got = tensor_hash(now[name]) if name in now else "absent"
            if got != h:
                raise RuntimeError(f"revert mismatch on {name}")


@torch.no_grad()
def apply_edit(model, edit: WeightEdit) -> Applied:
    params = dict(model.named_parameters())
    backup, created, new = {}, [], {}

    def touch(name):
        if name not in backup and name not in new:
            backup[name] = params[name].detach().clone()
            edit.before[name] = tensor_hash(params[name])
        return params[name]

    for op in edit.ops:
        kind = op["op"]
        if kind == "bias_add":
            mod = _module(model, op["module"])
            name = op["module"] + ".bias"
            v = _vec(op["vec"], mod.weight) * op["scale"]
            if mod.bias is None:
                edit.before[name] = "absent"
                mod.bias = torch.nn.Parameter(v.to(mod.weight.dtype), requires_grad=False)
                created.append(op["module"])
                new[name] = True
                params = dict(model.named_parameters())
            else:
                w = touch(name)
                w.copy_((w.float() + v).to(w.dtype))
        elif kind == "ortho":
            d = _vec(op["vec"], next(iter(params.values())))
            d = d / d.norm()
            for name in expand(model, op["tensors"]):
                w = touch(name)
                wf = w.float()
                w.copy_((wf - op.get("strength", 1.0) * torch.outer(d, d @ wf)).to(w.dtype))
        elif kind == "scale_cols":
            w = touch(op["tensor"])
            w[:, op["start"]:op["end"]] *= op["scale"]
        elif kind == "lowrank_add":
            w = touch(op["tensor"])
            a, b = _vec(op["A"], w), _vec(op["B"], w)
            w.copy_((w.float() + op["scale"] * (b @ a)).to(w.dtype))
        elif kind == "task_vector":
            from safetensors.torch import load_file

            from tr.policy import MODELS, snapshot
            path, _ = snapshot(*MODELS[op["other"]][:2])
            other = load_file(str(path / "model.safetensors"))
            for name in list(params):
                if name in other and params[name].is_floating_point() and other[name].shape == params[name].shape:
                    w = touch(name)
                    wf = w.float()
                    w.copy_((wf + op["lam"] * (wf - other[name].float().to(w.device))).to(w.dtype))
            del other
        else:
            raise ValueError(kind)
    params = dict(model.named_parameters())
    edit.tensors = sorted(edit.before)
    edit.after = {n: tensor_hash(params[n]) for n in edit.tensors}
    return Applied(model, edit, backup, created)


# ---- Adapter (LoRA form, hand-built) -------------------------------------------------------------------------------

def adapter_edit(adapter: Adapter, outdir) -> WeightEdit:
    """Merging an adapter is a WeightEdit of lowrank_add ops; A and B are saved as small .npy files."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    ops = []
    for i, (name, (a, b)) in enumerate(adapter.targets.items()):
        pa, pb = outdir / f"{adapter.name}.{i}.A.npy", outdir / f"{adapter.name}.{i}.B.npy"
        np.save(pa, np.asarray(a, dtype=np.float32))
        np.save(pb, np.asarray(b, dtype=np.float32))
        ops.append({"op": "lowrank_add", "tensor": name, "A": str(pa), "B": str(pb), "scale": adapter.alpha / adapter.rank})
    return WeightEdit(adapter.name, "lowrank", f"hand-built LoRA rank {adapter.rank} on {len(ops)} tensors", ops)


def attach(model, adapter: Adapter):
    """Unmerged form: side path y += (alpha/r) * B A x via forward hooks. Returns a detach function."""
    hooks = []
    for name, (a, b) in adapter.targets.items():
        mod = _module(model, name.rsplit(".", 1)[0])
        at = torch.as_tensor(np.asarray(a), dtype=mod.weight.dtype, device=mod.weight.device)
        bt = torch.as_tensor(np.asarray(b), dtype=mod.weight.dtype, device=mod.weight.device)
        s = adapter.alpha / adapter.rank
        hooks.append(mod.register_forward_hook(lambda m, inp, out, at=at, bt=bt, s=s: out + s * (inp[0] @ at.T) @ bt.T))
    return lambda: [h.remove() for h in hooks]
