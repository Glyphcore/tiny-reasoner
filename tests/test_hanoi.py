from tr.games.hanoi import Hanoi, HanoiVerifier, solve

G, V = Hanoi(), HanoiVerifier()


def play(state, moves):
    for m in moves:
        assert V.legal(state, m)
        state = G.apply(state, m)
    return state


def test_solve_every_level():
    for n in G.levels:
        s = G.new(n, 0)
        moves = solve(s)
        assert len(moves) == 2 ** n - 1 == G.optimal_steps(s)
        assert V.solved(play(s, moves))
        assert not V.solved(s)


def test_parse_format_roundtrip_and_garbage():
    s = G.new(3, 0)
    for _ in range(4):
        for m in G.legal_moves(s):
            assert G.parse(G.format(m)) == m
        s = G.apply(s, G.legal_moves(s)[0])
    assert G.parse("a->c") == ("A", "C") and G.parse("B to A") == ("B", "A") and G.parse("Move c b") == ("C", "B")
    assert G.parse("I think we should hmm") is None and G.parse("") is None


def test_optimal_increases_and_codec():
    opt = [G.optimal_steps(G.new(n, 0)) for n in G.levels]
    assert all(a < b for a, b in zip(opt, opt[1:]))
    s = play(G.new(4, 0), solve(G.new(4, 0))[:5])
    assert G.decode(G.encode(s)) == s


def test_mid_game_and_illegal():
    s = G.new(4, 0)
    s = play(s, solve(s)[:3])
    rest = solve(s)
    assert len(rest) == G.optimal_steps(s) == 12
    assert V.solved(play(s, rest))
    assert not V.legal(s, ("A", "A")) and "empty" in V.explain(G.new(2, 0), ("B", "C"))
    try:
        G.apply(G.new(2, 0), ("B", "C"))
        assert False
    except ValueError:
        pass
