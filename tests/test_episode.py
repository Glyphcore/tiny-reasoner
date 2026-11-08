import json
import math

import pytest

from spec.interfaces import ChatSuite, EpisodeConfig, GameSuite, Generation, PolicyConfig, RunConfig, TextSuite
from tests.fake_policy import FakePolicy, bad_then_good_policy, garbage_policy, perfect_policy
from tr.episode import GameEpisode
from tr.eval import Evaluator
from tr.games import get_game, names
from tr.run import load_run, new_run


def _cfg(suites=(), text=None, chats=None):
    return RunConfig(PolicyConfig("fake", "0", "fake"), list(suites), text or TextSuite("x"), chats or ChatSuite("y"))


@pytest.mark.parametrize("name", ["hanoi", "puzzle8", "countdown"])
def test_perfect_solo_and_chat(name):
    g, v = get_game(name)
    lv = g.levels[0]
    pol = perfect_policy(g)
    t = GameEpisode(g, v, pol, EpisodeConfig()).play(lv, 0)
    assert t.solved and len(t.steps) == t.optimal_steps and all(s.legal for s in t.steps)
    t = GameEpisode(g, v, pol, EpisodeConfig(mode="chat")).play(lv, 0)
    assert t.solved and len(t.steps) == t.optimal_steps


@pytest.mark.parametrize("name", ["hanoi", "puzzle8", "countdown"])
def test_chat_pushback(name):
    g, v = get_game(name)
    ep = GameEpisode(g, v, bad_then_good_policy(g, 2, v), EpisodeConfig(mode="chat"))
    t = ep.play(g.levels[0], 0)
    assert t.solved and t.n_illegal + t.n_unparseable == 2
    users = [m.content for m in ep.messages() if m.role == "user"]
    assert any("Try again." in u for u in users)  # illegal pushback with explain text
    assert any("could not find a move" in u for u in users)


@pytest.mark.parametrize("name", ["hanoi", "puzzle8", "countdown"])
def test_garbage_ends_unsolved(name):
    g, v = get_game(name)
    for mode in ("solo", "chat"):
        t = GameEpisode(g, v, garbage_policy(), EpisodeConfig(mode=mode, max_illegal=3)).play(g.levels[0], 0)
        assert not t.solved and len(t.steps) == 3 and t.steps[-1].error == "unparseable" and t.n_unparseable == 3


def test_history_bounded():
    g, v = get_game("hanoi")
    seen = []
    inner = perfect_policy(g)

    def nxt(msgs):
        seen.append(len(msgs))
        return inner.next_move(msgs)

    GameEpisode(g, v, FakePolicy(nxt), EpisodeConfig(mode="chat")).play(5, 0)
    assert max(seen) <= 25


def test_evaluator_games_and_run(tmp_path):
    g, v = get_game("hanoi")
    suite = GameSuite("hanoi", [2, 3], [0, 1, 2])
    run = new_run(_cfg([suite]), root=str(tmp_path))
    ms = Evaluator(run).games(perfect_policy(g), suite)
    assert [m.solve_rate for m in ms] == [1.0, 1.0] and all(m.legal_rate == 1.0 for m in ms)
    assert ms[0].mean_steps == ms[0].mean_optimal == 3
    assert len(list((tmp_path / run.id / "traces").glob("*.json"))) == 6
    lines = (tmp_path / run.id / "metrics.jsonl").read_text().splitlines()
    assert len(lines) == 8 and json.loads(lines[-1])["kind"] == "level"


def test_text_chats_summary_roundtrip(tmp_path):
    txt = tmp_path / "t.txt"
    txt.write_text("one two three\nfour five\nsix\n")
    chats = tmp_path / "c.json"
    chats.write_text(json.dumps([{"id": "a", "turns": ["hi", "again"]}, {"id": "b", "turns": ["yo"]}]))
    words = " ".join(f"w{i}" for i in range(10))
    rep = FakePolicy(lambda m: " ".join([words] * 5))  # 50 tokens, 10-gram cycle: 20-grams repeat
    cfg = _cfg([GameSuite("hanoi", [2], [0])], TextSuite(str(txt), 3), ChatSuite(str(chats), 2))
    run = new_run(cfg, "rt", root=str(tmp_path))
    ev = Evaluator(run)
    tm = ev.text(rep, cfg.text)
    assert tm.n_docs == 3 and tm.n_tokens == 6 and math.isfinite(tm.ppl)
    cm = ev.chats(rep, cfg.chats)
    assert cm.n_chats == 2 and cm.repeat_rate == 1.0 and cm.mean_reply_tokens == 50
    assert (tmp_path / "rt" / "chats" / "a.json").exists()
    uniq = FakePolicy(lambda m: " ".join(f"u{len(m)}_{i}" for i in range(50)))
    assert ev.chats(uniq, cfg.chats).repeat_rate == 0.0
    r = ev.all(perfect_policy(get_game("hanoi")[0], fallback="fine thanks"), cfg.games, cfg.text, cfg.chats)
    s = (tmp_path / "rt" / "summary.txt").read_text()
    assert s.count("\n") == 1 and s.startswith("variant=fake games: hanoi L2 solve=1.00") and "ppl=" in s
    assert len(r.game_chats) == 1
    back = load_run("rt", root=str(tmp_path))
    assert back.config == cfg or back.config.policy == cfg.policy
    assert back.config.games[0].levels == [2]
