"""Behaviour tests only: games end verified, evals are stable, edits and LoRA round-trip to exact hashes."""
import numpy as np
import pytest
import torch

from spec.interfaces import Adapter, WeightEdit
from tr.edits import adapter_edit, apply_edit, attach, model_hash
from tr.evalh import perplexity, play, probe, probe_states
from tr.games import GAMES
from tr.policy import CACHE, HFPolicy


class Oracle:
    """Scripted policy: plays an optimal move, except one illegal reply first when asked to (tests pushback)."""
    model_id = revision = "oracle"

    def __init__(self, game, seed, level, stumble=False):
        self.game, self.state, self.stumble = game, game.reset(seed, level), stumble

    def reply(self, messages, max_new_tokens):
        if self.stumble:
            self.stumble = False
            return "THINK: hmm.\nMOVE: banana", 5
        m = self.game.optimal_moves(self.state)[0]
        self.state = self.game.apply(self.state, m)
        return f"THINK: best move.\nMOVE: {m.text}", 5


@pytest.mark.parametrize("name", list(GAMES))
def test_game_plays_to_verified_end(name):
    game = GAMES[name]
    for seed, level in [(1, 2), (2, 3), (3, 4)]:
        ep = play(game, Oracle(game, seed, level, stumble=seed == 2), seed, level)
        assert ep.solved and ep.verified and len(ep.moves) == level
        assert sum(not t.legal for t in ep.traces) == (1 if seed == 2 else 0)
    for seed in range(5):
        s = game.reset(seed, 3)
        for m in game.all_moves(s):
            if m not in game.legal_moves(s):
                with pytest.raises(ValueError):
                    game.apply(s, m)


def tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, head_dim=8, max_position_embeddings=64)
    return Qwen3ForCausalLM(cfg).eval()


def test_weight_edit_and_revert_restore_exact_hashes(tmp_path):
    model = tiny()
    base = model_hash(model)
    v = tmp_path / "v.npy"
    np.save(v, np.random.default_rng(0).normal(size=32).astype(np.float32))
    ops = [{"op": "bias_add", "module": "model.layers.0.mlp.down_proj", "vec": str(v), "scale": 2.0},
           {"op": "ortho", "tensors": [r"model\.layers\.\d+\.mlp\.down_proj\.weight"], "vec": str(v)},
           {"op": "scale_cols", "tensor": "model.layers.1.self_attn.o_proj.weight", "start": 8, "end": 16, "scale": 0.0}]
    edit = WeightEdit("t", "mixed", "test", ops)
    applied = apply_edit(model, edit)
    assert model_hash(model) != base
    assert edit.before["model.layers.0.mlp.down_proj.bias"] == "absent"
    assert all(edit.before[n] != edit.after[n] for n in edit.tensors)
    d = torch.from_numpy(np.load(v))
    assert float((d @ model.model.layers[1].mlp.down_proj.weight.float()).abs().max()) < 1e-4
    applied.revert()
    assert model_hash(model) == base and model.model.layers[0].mlp.down_proj.bias is None


def test_lora_round_trip(tmp_path):
    model = tiny()
    base = model_hash(model)
    rng = np.random.default_rng(1)
    name = "model.layers.1.mlp.down_proj.weight"
    ad = Adapter("lora-t", 2, 4.0, {name: (rng.normal(size=(2, 64)).astype(np.float32) * 0.1,
                                           rng.normal(size=(32, 2)).astype(np.float32) * 0.1)})
    ids = torch.arange(10)[None]
    with torch.no_grad():
        y0 = model(ids).logits
        detach = attach(model, ad)
        y_side = model(ids).logits
        detach()
        assert model_hash(model) == base
        applied = apply_edit(model, adapter_edit(ad, tmp_path))
        y_merged = model(ids).logits
    assert not torch.allclose(y0, y_merged, atol=1e-5)
    assert torch.allclose(y_side, y_merged, atol=1e-4)  # unmerged side path == merged weights
    applied.revert()
    assert model_hash(model) == base


@pytest.mark.skipif(not CACHE.exists(), reason="model not downloaded")
def test_eval_is_stable_on_fixed_seeds():
    policy = HFPolicy()
    game = GAMES["hanoi"]

    def once():
        policy.reset_cache()
        ep = play(game, policy, 300, 2)
        pr = probe(game, policy, probe_states(game, "smoke", 4))
        return [(t.text, t.legal) for t in ep.traces], ep.solved, ep.verified, pr, perplexity(policy)
    a, b = once(), once()
    assert a == b and a[2]
