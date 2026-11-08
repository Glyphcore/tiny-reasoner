"""Sliding 8-puzzle. State: tuple of 9 ints row-major, 0 = blank. Action: tile number (1-8) sliding into the blank."""
from __future__ import annotations

import pickle
import random
import re
from collections import deque
from pathlib import Path
from typing import Mapping, Optional, Sequence

from spec.interfaces import Action, Game, State, Verifier

GOAL: State = (1, 2, 3, 4, 5, 6, 7, 8, 0)
BUCKETS = {1: (2, 4), 2: (5, 8), 3: (9, 14), 4: (15, 20)}
_CACHE = Path(__file__).resolve().parents[2] / ".cache" / "puzzle8_dist.pkl"
_dist: Optional[dict] = None


def _neighbours(state: State):
    b = state.index(0)
    r, c = divmod(b, 3)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < 3 and 0 <= nc < 3:
            j = nr * 3 + nc
            s = list(state)
            s[b], s[j] = s[j], 0
            yield state[j], tuple(s)


def distances() -> dict:
    """BFS distance from the goal for all 181440 reachable states (lazy, cached)."""
    global _dist
    if _dist is not None:
        return _dist
    if _CACHE.exists():
        with open(_CACHE, "rb") as f:
            _dist = pickle.load(f)
        return _dist
    d = {GOAL: 0}
    q = deque([GOAL])
    while q:
        s = q.popleft()
        for _, n in _neighbours(s):
            if n not in d:
                d[n] = d[s] + 1
                q.append(n)
    _dist = d
    if _CACHE.parent.is_dir():
        with open(_CACHE, "wb") as f:
            pickle.dump(d, f)
    return d


def solve(state: State) -> list[Action]:
    """Optimal solution: tiles to slide, in order."""
    d = distances()
    state = tuple(state)
    out: list[Action] = []
    while state != GOAL:
        tile, state = min(_neighbours(state), key=lambda t: d[t[1]])
        out.append(tile)
    return out


class Puzzle8(Game):
    name = "puzzle8"
    levels = [1, 2, 3, 4]

    def new(self, level: int, seed: int) -> State:
        lo, hi = BUCKETS[level]
        d = distances()
        rng = random.Random(seed)
        state, prev = GOAL, None
        while True:
            opts = [(t, n) for t, n in _neighbours(state) if n != prev]
            _, nxt = opts[rng.randrange(len(opts))]
            prev, state = state, nxt
            if lo <= d[state] <= hi:
                return state

    def legal_moves(self, state: State) -> Sequence[Action]:
        return sorted(t for t, _ in _neighbours(tuple(state)))

    def apply(self, state: State, action: Action) -> State:
        for t, n in _neighbours(tuple(state)):
            if t == action:
                return n
        raise ValueError(f"illegal move: tile {action} is not next to the blank")

    def optimal_steps(self, state: State) -> int:
        return distances()[tuple(state)]

    def render(self, state: State) -> str:
        def row(r):
            return " ".join("_" if v == 0 else str(v) for v in r)
        def grid(s):
            return "\n".join(row(s[i:i + 3]) for i in (0, 3, 6))
        return grid(state) + "\nGoal:\n" + grid(GOAL)

    def rules(self) -> str:
        return ("8-puzzle: a 3x3 grid holds tiles 1-8 and one blank (_). Each move slides one tile that is next to the blank "
                "(up, down, left or right) into the blank. Reach the goal grid shown. "
                "Reply with exactly one move as 'move N' where N is the tile number (1-8), for example 'move 5'.")

    def parse(self, text: str) -> Optional[Action]:
        m = re.search(r"\d+", text or "")
        if not m:
            return None
        n = int(m.group())
        return n if 1 <= n <= 8 else None

    def format(self, action: Action) -> str:
        return f"move {action}"

    def encode(self, state: State) -> Mapping:
        return {"tiles": list(state)}

    def decode(self, data: Mapping) -> State:
        return tuple(int(x) for x in data["tiles"])


class Puzzle8Verifier(Verifier):
    def __init__(self, game: Optional[Puzzle8] = None):
        self.game = game or Puzzle8()

    def legal(self, state: State, action: Action) -> bool:
        return action in self.game.legal_moves(state)

    def solved(self, state: State) -> bool:
        return tuple(state) == GOAL

    def explain(self, state: State, action: Action) -> str:
        if not isinstance(action, int) or not 1 <= action <= 8:
            return f"{action!r} is not a tile; tiles are numbered 1 to 8"
        return f"tile {action} is not next to the blank"
