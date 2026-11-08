"""Recipe edits: a WeightEdit described by a small JSON recipe (plus optional auxiliary tensors), rebuilt tensor by tensor.

Unlike TensorEdit this never holds whole-model copies: `apply` computes each new tensor from the live one (and the
recipe), `revert` streams the original tensors back from the model's own safetensors checkpoint on disk, so the revert is
bit-exact and RAM stays flat even for edits touching every layer (task vectors).

Recipe kinds (all scales are floats, layers are ints, tensor names are the HF parameter names):
  taskvec   {"other": "<hf id>", "other_revision": str, "lam": float, "include": ["self_attn", "mlp"]}
            W = W + lam * (W_other - W)   (lam > 0 extrapolates away from `other`, lam < 0 interpolates towards it)
  steer     {"layer": L, "where": "o_proj"|"down_proj", "dir": "<key in aux>", "mean_input": "<key in aux>", "scale": s}
            bake a steering vector: W += s * dir (x) m/|m|^2   (adds s*dir on the mean input)
  orth      {"layers": [..], "dir": "<key>"}   remove dir from o_proj and down_proj rows (and embed_tokens if "embed": true)
  headscale {"layer": L, "heads": [..], "factor": f}
  neuron    {"layer": L, "idx": [..], "factor": f}          scale down_proj columns
  lora      {"targets": [{"tensor": name, "A": "<key>", "B": "<key>", "alpha": a}, ...]}
Aux tensors live in <path>.safetensors; the recipe and hashes in <path>.json.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from typing import Mapping, Optional

from spec.interfaces import JSON, Policy, Tensor, TensorChange, WeightEdit

from tr.edits import ops
from tr.edits.weightedit import tensor_hash


def checkpoint_file(model_id: str, revision: str) -> str:
    from huggingface_hub import snapshot_download
    p = snapshot_download(model_id, revision=revision, local_files_only=True, allow_patterns=["*.safetensors", "*.json"])
    files = [f for f in os.listdir(p) if f.endswith(".safetensors")]
    assert len(files) == 1, f"expected one safetensors shard in {p}, got {files}"
    return os.path.join(p, files[0])


class RecipeEdit(WeightEdit):
    def __init__(self, id: str, recipe: JSON, aux: Optional[Mapping[str, Tensor]] = None, changes=None,
                 base_file: Optional[str] = None):
        self.id = id
        self.base_file = base_file  # safetensors file with the original tensors; default: the model's HF checkpoint
        self.kind = recipe["kind"]
        self.recipe = dict(recipe)
        self.aux = dict(aux or {})
        self.changes: list[TensorChange] = list(changes or [])
        self.applied = False

    # ---- which tensors, and the new value of each
    def _targets(self, policy: Policy) -> dict[str, JSON]:
        r = self.recipe
        k = self.kind
        if k == "taskvec":
            inc = tuple(r.get("include", ["self_attn", "mlp"]))
            return {n: {"op": "taskvec", "lam": r["lam"]} for n in policy.tensor_names()
                    if any(f".{i}." in n for i in inc) and n.endswith("weight") and "norm" not in n}
        if k == "steer":
            where = "self_attn.o_proj" if r["where"] == "o_proj" else "mlp.down_proj"
            return {f"model.layers.{r['layer']}.{where}.weight": {"op": "steer", "scale": r["scale"], "dir": r["dir"]}}
        if k == "orth":
            out = {}
            for L in r["layers"]:
                out[f"model.layers.{L}.self_attn.o_proj.weight"] = {"op": "orth_rows", "dir": r["dir"]}
                out[f"model.layers.{L}.mlp.down_proj.weight"] = {"op": "orth_rows", "dir": r["dir"]}
            if r.get("embed"):
                out["model.embed_tokens.weight"] = {"op": "orth_cols", "dir": r["dir"]}
            return out
        if k == "headscale":
            return {f"model.layers.{r['layer']}.self_attn.o_proj.weight": {"op": "headscale", "heads": r["heads"], "factor": r["factor"]}}
        if k == "neuron":
            return {f"model.layers.{r['layer']}.mlp.down_proj.weight": {"op": "scale_cols", "idx": r["idx"], "factor": r["factor"]}}
        if k == "lora":
            return {t["tensor"]: {"op": "lora", "A": t["A"], "B": t["B"], "alpha": t["alpha"]} for t in r["targets"]}
        raise ValueError(f"unknown recipe kind {k}")

    def _new_value(self, name: str, w, how: JSON, other=None):
        op = how["op"]
        if op == "taskvec":
            return ops.task_vector(w, other.get_tensor(name).float(), how["lam"])
        if op == "steer":
            return ops.bake_steer(w, self.aux[how["dir"]].float(), self.aux[self.recipe["mean_input"]].float(), how["scale"])
        if op == "orth_rows":
            return ops.orthogonalize_rows(w, self.aux[how["dir"]].float())
        if op == "orth_cols":
            return ops.orthogonalize_cols(w, self.aux[how["dir"]].float())
        if op == "headscale":
            return ops.scale_heads(w, how["heads"], how["factor"])
        if op == "scale_cols":
            return ops.scale_cols(w, how["idx"], how["factor"])
        if op == "lora":
            A, B = self.aux[how["A"]].float(), self.aux[how["B"]].float()
            return w + how["alpha"] * (B.to(w) @ A.to(w))
        raise ValueError(op)

    # ---- WeightEdit
    def apply(self, policy: Policy) -> None:
        import torch
        from safetensors import safe_open
        targets = self._targets(policy)
        first = not self.changes
        if not first and not self.verify(policy, "before"):
            raise RuntimeError(f"{self.id}: live tensors do not match hash_before; refusing to apply")
        other = None
        if self.kind == "taskvec":
            other = safe_open(checkpoint_file(self.recipe["other"], self.recipe["other_revision"]), "pt", device="cpu")
        changes = []
        with torch.no_grad():
            for name in sorted(targets):
                p = policy.tensor(name)
                hb = tensor_hash(p)
                new = self._new_value(name, p.detach().float(), targets[name], other).to(p.dtype)
                p.copy_(new)
                changes.append(TensorChange(name=name, hash_before=hb, hash_after=tensor_hash(p), how=dict(targets[name])))
        if first:
            self.changes = changes
        elif [c.hash_after for c in changes] != [c.hash_after for c in self.changes]:
            raise RuntimeError(f"{self.id}: rebuilt edit does not reproduce the recorded hash_after")
        self.applied = True

    def revert(self, policy: Policy) -> None:
        import torch
        from safetensors import safe_open
        base = self.base_file or checkpoint_file(policy.config.model, policy.config.revision)
        f = safe_open(base, "pt", device="cpu")
        with torch.no_grad():
            for c in self.changes:
                p = policy.tensor(c.name)
                p.copy_(f.get_tensor(c.name).to(p.dtype))
        self.applied = False
        if not self.verify(policy, "before"):
            raise RuntimeError(f"{self.id}: revert did not restore hash_before")

    def verify(self, policy: Policy, expect: str) -> bool:
        key = "hash_before" if expect == "before" else "hash_after"
        return bool(self.changes) and all(tensor_hash(policy.tensor(c.name)) == getattr(c, key) for c in self.changes)

    def save(self, path: str) -> None:
        from safetensors.torch import save_file
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {"id": self.id, "kind": self.kind, "recipe": self.recipe, "changes": [asdict(c) for c in self.changes]}
        with open(path + ".json", "w") as f:
            json.dump(meta, f, indent=1)
        if self.aux:
            save_file({k: v.detach().cpu().contiguous() for k, v in self.aux.items()}, path + ".safetensors")

    def load(self, path: str) -> "RecipeEdit":
        return load_recipe(path)


def load_recipe(path: str) -> RecipeEdit:
    from safetensors.torch import load_file
    with open(path + ".json") as f:
        meta = json.load(f)
    aux = load_file(path + ".safetensors") if os.path.exists(path + ".safetensors") else {}
    changes = [TensorChange(**c) for c in meta["changes"]]
    return RecipeEdit(meta["id"], meta["recipe"], aux, changes)


def load_any(path: str) -> WeightEdit:
    """Load a saved edit of either class (TensorEdit json has no 'recipe' key)."""
    with open(path + ".json") as f:
        meta = json.load(f)
    if "recipe" in meta:
        return load_recipe(path)
    from tr.edits.weightedit import load_edit
    return load_edit(path)
