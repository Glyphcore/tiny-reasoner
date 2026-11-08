"""Game registry. Modules are imported lazily."""
from __future__ import annotations

import importlib

_MODULES = {
    "hanoi": ("tr.games.hanoi", "Hanoi", "HanoiVerifier"),
    "puzzle8": ("tr.games.puzzle8", "Puzzle8", "Puzzle8Verifier"),
    "countdown": ("tr.games.countdown", "Countdown", "CountdownVerifier"),
}


def names() -> list[str]:
    return list(_MODULES)


def _classes(name: str):
    if name not in _MODULES:
        raise KeyError(f"unknown game {name!r}; known: {names()}")
    mod, g, v = _MODULES[name]
    m = importlib.import_module(mod)
    return getattr(m, g), getattr(m, v)


class _Games:
    def __getitem__(self, name):
        return _classes(name)

    def __iter__(self):
        return iter(_MODULES)

    def __len__(self):
        return len(_MODULES)

    def keys(self):
        return _MODULES.keys()


GAMES = _Games()  # name -> (GameClass, VerifierClass), resolved on access


def get_game(name: str):
    g, v = _classes(name)
    game = g()
    return game, v(game)
