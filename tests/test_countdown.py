from statistics import mean

import pytest

from tr.games.countdown import Countdown, CountdownVerifier, solve

G = Countdown()
V = CountdownVerifier(G)
SEEDS = range(10)


@pytest.mark.parametrize("level", G.levels)
def test_instances_solve_optimally(level):
    for seed in SEEDS:
        st = G.new(level, seed)
        assert st == G.new(level, seed)
        nums, target = st
        assert len(nums) == level + 2 and target not in nums and not V.solved(st)
        opt = G.optimal_steps(st)
        assert opt >= (1 if level == 1 else 2)
        plan = solve(st)
        assert len(plan) == opt
        for a in plan:
            assert V.legal(st, a)
            st = G.apply(st, a)
        assert V.solved(st)
        assert G.decode(G.encode(G.new(level, seed))) == G.new(level, seed)


def test_level_difficulty_monotone():
    m = [mean(G.optimal_steps(G.new(l, s)) for s in SEEDS) for l in G.levels]
    assert m == sorted(m)


def test_parse_format_and_verifier():
    assert G.parse("I play 7 * 3 = 21") == (7, "*", 3)
    assert G.parse("25 x 4") == (25, "*", 4)
    assert G.parse("10 ÷ 2") == (10, "/", 2)
    assert G.parse("no move here") is None
    for a in [(3, "+", 4), (9, "-", 2), (6, "/", 3)]:
        assert G.parse(G.format(a)) == a
    st = ((2, 3, 9), 50)
    assert not V.legal(st, (7, "+", 3)) and V.explain(st, (7, "+", 3)) == "7 is not available"
    assert V.explain(st, (9, "/", 2)) == "9 / 2 is not a whole number"
    assert not V.legal(st, (2, "-", 3))
