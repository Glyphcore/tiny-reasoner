"""Weight edit + revert restores exact hashes; LoRA merge round-trips through disk. CPU, tiny model, no HF download."""
import pytest

torch = pytest.importorskip("torch")

from spec.interfaces import LoraTarget, PolicyConfig
from tr.edits.lora import LoraAdapter, load_adapter
from tr.edits.weightedit import load_edit, make_edit, tensor_hash


class TinyPolicy:
    def __init__(self):
        torch.manual_seed(0)
        self.model = torch.nn.Sequential(torch.nn.Linear(8, 16, bias=False), torch.nn.Linear(16, 8, bias=True)).half()
        self.tokenizer = None
        self.config = PolicyConfig(model="tiny", revision="0", variant="base", device="cpu")

    def generate(self, messages): raise NotImplementedError
    def nll(self, text): raise NotImplementedError
    def tensor(self, name): return dict(self.model.named_parameters())[name]
    def tensor_names(self): return [n for n, _ in self.model.named_parameters()]


def test_edit_apply_revert_exact(tmp_path):
    pol = TinyPolicy()
    names = ["0.weight", "1.bias"]
    h0 = {n: tensor_hash(pol.tensor(n)) for n in names}
    edit = make_edit("scale0", "head_scale", pol, {
        "0.weight": (lambda w: w * 0.5, {"op": "scale", "factor": 0.5}),
        "1.bias": (lambda b: b + 1.0, {"op": "add", "value": 1.0}),
    })
    assert edit.verify(pol, "before") and not edit.applied
    edit.apply(pol)
    assert edit.applied and edit.verify(pol, "after") and not edit.verify(pol, "before")
    assert {n: tensor_hash(pol.tensor(n)) for n in names} != h0
    with pytest.raises(RuntimeError):
        edit.apply(pol)  # live tensors no longer match hash_before
    # save while applied, reload, revert through the reloaded copy
    edit.save(str(tmp_path / "e"))
    edit2 = load_edit(str(tmp_path / "e"))
    assert [c.hash_after for c in edit2.changes] == [c.hash_after for c in edit.changes]
    edit2.revert(pol)
    assert {n: tensor_hash(pol.tensor(n)) for n in names} == h0
    assert edit2.verify(pol, "before")
    # re-apply from disk works on the restored base
    edit2.apply(pol)
    assert edit2.verify(pol, "after")


def test_lora_merge_round_trip(tmp_path):
    pol = TinyPolicy()
    h0 = tensor_hash(pol.tensor("0.weight"))
    A = torch.randn(2, 8) * 0.1
    B = torch.randn(16, 2) * 0.1
    ad = LoraAdapter("hand1", [LoraTarget(tensor="0.weight", A=A, B=B, alpha=2.0)])
    ad.save(str(tmp_path / "a"))
    ad2 = load_adapter(str(tmp_path / "a"))
    assert ad2.rank == 2 and ad2.targets[0].tensor == "0.weight"
    assert torch.allclose(ad2.targets[0].A, A) and torch.allclose(ad2.targets[0].B, B)
    w_before = pol.tensor("0.weight").detach().float().clone()
    edit = ad2.merge(pol)
    assert edit.kind == "lora_merge" and edit.changes[0].how["rank"] == 2
    edit.apply(pol)
    w_after = pol.tensor("0.weight").detach().float()
    assert torch.allclose(w_after, (w_before + 2.0 * B @ A).half().float())
    edit.revert(pol)
    assert tensor_hash(pol.tensor("0.weight")) == h0
