import math

import pytest

from spec.interfaces import Message


@pytest.mark.gpu
def test_policy_generates_scores_and_exposes_tensors():
    from tr.policy import default_config, load_policy
    pol = load_policy(default_config("base", thinking=False, max_new_tokens=32))
    g = pol.generate([Message("user", "What is 2+3? Answer with one number.")])
    assert "5" in g.text and g.reasoning == "" and not g.truncated
    s, n = pol.nll("The cat sat on the mat and looked out of the window.")
    assert math.isfinite(s) and s > 0 and n > 0
    assert tuple(pol.tensor("model.layers.0.mlp.down_proj.weight").shape) == (1024, 3072)
    assert "model.embed_tokens.weight" in pol.tensor_names()
