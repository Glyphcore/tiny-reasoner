"""Scripted policies for tests: no torch, no model."""
from __future__ import annotations

from spec.interfaces import Generation, Policy, PolicyConfig
from tr.games import get_game
import importlib


class FakePolicy(Policy):
    """next_move(messages) -> reply text (str) or Generation."""

    def __init__(self, next_move, variant="fake", thinking=False):
        self.config = PolicyConfig(model="fake", revision="0", variant=variant, thinking=thinking)
        self.model = None
        self.tokenizer = None
        self.next_move = next_move
        self.calls = 0

    def generate(self, messages):
        self.calls += 1
        r = self.next_move(list(messages))
        if isinstance(r, Generation):
            return r
        return Generation(text=r, reasoning="", n_tokens=len(r.split()), truncated=False)

    def nll(self, text):
        return (len(text) * 0.5, len(text.split()) or 1)

    def tensor(self, name):
        raise KeyError(name)

    def tensor_names(self):
        return []


def _solve_fn(game):
    return importlib.import_module(f"tr.games.{game.name}").solve


def _state_cache(game):
    cache = {}
    for lv in game.levels:
        for sd in range(20):
            s = game.new(lv, sd)
            cache[game.render(s)] = s
    return cache


def perfect_policy(game, fallback=None, **kw) -> FakePolicy:
    """Finds the current state by matching the latest rendered board in the messages, then plays solve()[0]."""
    solve, cache = _solve_fn(game), _state_cache(game)

    def nxt(messages):
        for m in reversed(messages):
            if m.role != "user":
                continue
            hits = [r for r in cache if r in m.content]
            if hits:
                state = cache[max(hits, key=len)]
                a = solve(state)[0]
                cache.setdefault(game.render(game.apply(state, a)), game.apply(state, a))
                return f"Thinking about it.\n{game.format(a)}"
        if fallback is not None:
            return fallback
        raise AssertionError("no board found in messages")

    return FakePolicy(nxt, **kw)


_ILLEGAL_TRIES = ["move A A", "move 0", "move 9", "99 + 99", "1 / 0", "move Z"]


def illegal_text(game, verifier, state) -> str:
    for t in _ILLEGAL_TRIES:
        a = game.parse(t)
        if a is not None and not verifier.legal(state, a):
            return t
    raise AssertionError("no illegal candidate for " + game.name)


def bad_then_good_policy(game, n_bad: int, verifier=None, **kw) -> FakePolicy:
    """First n_bad replies are bad (illegal when possible on the first, else unparseable), then perfect play."""
    good = perfect_policy(game, **kw)
    left = [n_bad]
    verifier = verifier or get_game(game.name)[1]

    def nxt(messages):
        if left[0] > 0:
            left[0] -= 1
            if left[0] == n_bad - 1:  # first bad reply: illegal if a candidate exists
                for m in messages:
                    pass
                state = None
                for m in reversed(messages):
                    if m.role == "user":
                        for r, s in _state_cache(game).items():
                            if r in m.content:
                                state = s
                                break
                        break
                if state is not None:
                    try:
                        return illegal_text(game, verifier, state)
                    except AssertionError:
                        pass
            return "Hmm, I am not sure what to do here."
        return good.next_move(messages)

    return FakePolicy(nxt, **kw)


def garbage_policy(**kw) -> FakePolicy:
    return FakePolicy(lambda m: "I refuse to play, banana.", **kw)
