"""Tower of Hanoi. State: three tuples of disk sizes (bottom to top) for pegs A, B, C. Action: (from_peg, to_peg) letters."""
from __future__ import annotations

import re
from typing import Optional, Sequence

from spec.interfaces import Game, Verifier

PEGS = "ABC"
# explicit forms accept any case ("move a c", "a->c", "B to A"); the bare "A C" form must be uppercase so prose does not match
_MOVE_RE = re.compile(r"\b(?:[Mm]ove\s+([A-Ca-c])\s*(?:->|[Tt]o|\s)\s*([A-Ca-c])|([A-Ca-c])\s*(?:->|\s[Tt]o\s)\s*([A-Ca-c])|([ABC])\s+([ABC]))\b")


def _plan(state, k: int, target: int, out: list) -> None:
    """Append moves that put disks 1..k on peg index `target`, starting from `state` (mutated in place as lists)."""
    if k == 0:
        return
    src = next(i for i in range(3) if k in state[i])
    if src == target:
        _plan(state, k - 1, target, out)
        return
    other = 3 - src - target
    _plan(state, k - 1, other, out)
    state[src].remove(k)
    state[target].append(k)
    out.append((PEGS[src], PEGS[target]))
    _plan(state, k - 1, target, out)


def solve(state) -> list:
    """Optimal move list from any legal state to all disks on C."""
    work = [list(p) for p in state]
    out: list = []
    _plan(work, sum(len(p) for p in work), 2, out)
    return out


class Hanoi(Game):
    name = "hanoi"
    levels = [2, 3, 4, 5]

    def new(self, level: int, seed: int = 0):
        return (tuple(range(level, 0, -1)), (), ())

    def _why(self, state, action) -> str:
        try:
            a, b = action
            i, j = PEGS.index(a), PEGS.index(b)
        except (ValueError, TypeError):
            return f"{action!r} is not a move between pegs A, B, C"
        if i == j:
            return "source and target peg are the same"
        if not state[i]:
            return f"peg {a} is empty"
        d = state[i][-1]
        if state[j] and state[j][-1] < d:
            return f"disk {d} cannot go on disk {state[j][-1]}"
        return ""

    def legal_moves(self, state) -> Sequence:
        return [(a, b) for a in PEGS for b in PEGS if not self._why(state, (a, b))]

    def apply(self, state, action):
        why = self._why(state, action)
        if why:
            raise ValueError(why)
        i, j = PEGS.index(action[0]), PEGS.index(action[1])
        pegs = [list(p) for p in state]
        pegs[j].append(pegs[i].pop())
        return tuple(tuple(p) for p in pegs)

    def optimal_steps(self, state) -> int:
        return len(solve(state))

    def render(self, state) -> str:
        lines = [f"{PEGS[i]}: {list(state[i])}" for i in range(3)]
        return "\n".join(lines + ["Goal: all disks on C"])

    def rules(self) -> str:
        return ("Tower of Hanoi with pegs A, B, C. Disks are numbered by size (1 is smallest) and listed bottom to top. "
                "Move one disk at a time: the top disk of one peg goes onto another peg, never onto a smaller disk. "
                "Goal: all disks on peg C. Reply with one move in the exact form `move A C` (from peg A to peg C).")

    def parse(self, text: str) -> Optional[tuple]:
        m = _MOVE_RE.search(text)
        if not m:
            return None
        a, b = [g for g in m.groups() if g is not None][:2]
        return (a.upper(), b.upper())

    def format(self, action) -> str:
        return f"move {action[0]} {action[1]}"

    def encode(self, state):
        return {"pegs": [list(p) for p in state]}

    def decode(self, data):
        return tuple(tuple(p) for p in data["pegs"])


class HanoiVerifier(Verifier):
    def __init__(self, game: Optional[Hanoi] = None):
        self.game = game or Hanoi()

    def legal(self, state, action) -> bool:
        return not self.game._why(state, action)

    def solved(self, state) -> bool:
        return not state[0] and not state[1] and len(state[2]) > 0

    def explain(self, state, action) -> str:
        return self.game._why(state, action) or "legal move"
