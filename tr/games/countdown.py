"""Countdown numbers game. State: (sorted numbers tuple, target). Action: (a, op, b) with op in + - * /."""
from __future__ import annotations

import random
import re
from functools import lru_cache
from typing import Optional, Sequence

from spec.interfaces import Game, Verifier

OPS = "+-*/"
_MOVE_RE = re.compile(r"(\d+)\s*([+\-*/x×÷])\s*(\d+)")
_NORM = {"x": "*", "×": "*", "÷": "/"}
_BIG = (25, 50, 75, 100)
_LEVEL_SIZE = {1: 3, 2: 4, 3: 5, 4: 6}
_TARGETS = {1: (10, 30), 2: (20, 60), 3: (50, 200), 4: (100, 999)}
_INF = 10 ** 6


def _combine(a: int, op: str, b: int) -> Optional[int]:
    if op == "+":
        return a + b
    if op == "-":
        r = a - b
    elif op == "*":
        return a * b
    elif op == "/":
        if b == 0 or a % b:
            return None
        r = a // b
    else:
        return None
    return r if r > 0 else None


def _results(nums: tuple):
    """Yield (a, op, b, new_numbers) for each distinct legal move."""
    seen = set()
    n = len(nums)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            a, b = nums[i], nums[j]
            for op in OPS:
                if op in "+*" and a > b:
                    continue
                r = _combine(a, op, b)
                if r is None or (a, op, b) in seen:
                    continue
                seen.add((a, op, b))
                rest = [nums[k] for k in range(n) if k != i and k != j]
                yield a, op, b, tuple(sorted(rest + [r]))


@lru_cache(maxsize=None)
def _best(nums: tuple, target: int) -> tuple:
    """(steps, first_move) minimal operations to get target into the pool; (_INF, None) if impossible."""
    if target in nums:
        return 0, None
    best, mv = _INF, None
    if len(nums) >= 2:
        for a, op, b, new in _results(nums):
            s = _best(new, target)[0] + 1
            if s < best:
                best, mv = s, (a, op, b)
    return best, mv


def solve(state) -> list:
    """Optimal action list from state to a pool containing the target."""
    nums, target = state
    out = []
    while _best(nums, target)[0] not in (0, _INF):
        a, op, b = _best(nums, target)[1]
        out.append((a, op, b))
        nums = _apply(nums, (a, op, b))
    return out


def _apply(nums: tuple, action) -> tuple:
    a, op, b = action
    pool = list(nums)
    for v in (a, b):
        if v not in pool:
            raise ValueError(f"{v} is not available")
        pool.remove(v)
    r = _combine(a, op, b)
    if r is None:
        raise ValueError(_why(a, op, b))
    return tuple(sorted(pool + [r]))


def _why(a: int, op: str, b: int) -> str:
    if op == "/" and (b == 0 or a % b):
        return f"{a} / {b} is not a whole number"
    if op == "-" and a <= b:
        return f"{a} - {b} is not positive"
    if op not in OPS:
        return f"unknown operation {op}"
    return f"{a} {op} {b} is not allowed"


class Countdown(Game):
    name = "countdown"
    levels = [1, 2, 3, 4]

    def new(self, level: int, seed: int):
        rng = random.Random(seed)
        k = _LEVEL_SIZE[level]
        lo, hi = _TARGETS[level]
        min_steps = 1 if level == 1 else 2
        while True:
            if level <= 2:
                nums = [rng.randint(1, 10) for _ in range(k)]
            else:
                nums = [rng.randint(1, 10) for _ in range(k - 1)] + [rng.choice(_BIG)]
            nums = tuple(sorted(nums))
            target = rng.randint(lo, hi)
            if target in nums:
                continue
            s = _best(nums, target)[0]
            if min_steps <= s < _INF:
                return (nums, target)

    def legal_moves(self, state) -> Sequence:
        return [(a, op, b) for a, op, b, _ in _results(state[0])]

    def apply(self, state, action):
        return (_apply(state[0], action), state[1])

    def optimal_steps(self, state) -> int:
        s = _best(state[0], state[1])[0]
        if s >= _INF:
            raise ValueError("unsolvable state")
        return s

    def render(self, state) -> str:
        return f"Numbers: {', '.join(map(str, state[0]))}   Target: {state[1]}"

    def rules(self) -> str:
        return ("Countdown: combine the given numbers with + - * / to make the target. Each move picks two numbers from the "
                "list and replaces them with the result, which must be a positive whole number. You win when the target is in "
                "the list. Reply with one move per turn in the form 'a op b', for example: 7 * 3")

    def parse(self, text: str):
        m = _MOVE_RE.search(text)
        if not m:
            return None
        op = _NORM.get(m.group(2), m.group(2))
        return (int(m.group(1)), op, int(m.group(3)))

    def format(self, action) -> str:
        a, op, b = action
        return f"{a} {op} {b}"

    def encode(self, state):
        return {"numbers": list(state[0]), "target": state[1]}

    def decode(self, data):
        return (tuple(sorted(int(x) for x in data["numbers"])), int(data["target"]))


class CountdownVerifier(Verifier):
    def __init__(self, game: Optional[Countdown] = None):
        self.game = game or Countdown()

    def legal(self, state, action) -> bool:
        try:
            _apply(state[0], action)
        except (ValueError, TypeError):
            return False
        return True

    def solved(self, state) -> bool:
        return state[1] in state[0]

    def explain(self, state, action) -> str:
        try:
            _apply(state[0], action)
        except ValueError as e:
            return str(e)
        except TypeError:
            return "malformed move"
        return "legal"
