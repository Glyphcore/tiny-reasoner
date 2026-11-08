from tr.games.puzzle8 import BUCKETS, GOAL, Puzzle8, Puzzle8Verifier, solve

G = Puzzle8()
V = Puzzle8Verifier(G)


def test_levels_solve_and_verify():
    means = []
    for lv in G.levels:
        lo, hi = BUCKETS[lv]
        opt = []
        for seed in range(5):
            s = G.new(lv, seed)
            assert s != GOAL and not V.solved(s)
            n = G.optimal_steps(s)
            assert lo <= n <= hi
            sol = solve(s)
            assert len(sol) == n
            for a in sol:
                assert V.legal(s, a)
                s = G.apply(s, a)
            assert V.solved(s)
            opt.append(n)
        means.append(sum(opt) / len(opt))
    assert all(a < b for a, b in zip(means, means[1:]))


def test_determinism_and_variety():
    for lv in G.levels:
        assert G.new(lv, 3) == G.new(lv, 3)
        assert len({G.new(lv, s) for s in range(5)}) >= 3


def test_parse_format_encode():
    s = G.new(2, 1)
    for a in G.legal_moves(s):
        assert G.parse(G.format(a)) == a
    assert G.parse("slide 5") == 5 and G.parse("tile 7") == 7 and G.parse("3") == 3
    for bad in ["", "hello", "move 0", "move 9", "move"]:
        assert G.parse(bad) is None
    assert G.decode(G.encode(s)) == s
    bad = next(t for t in range(1, 9) if t not in G.legal_moves(s))
    assert not V.legal(s, bad) and "not next to the blank" in V.explain(s, bad)
