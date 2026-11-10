"""Three games with exact verifiers. States are hashable tuples; level = optimal solution length (BFS)."""
from __future__ import annotations

import random
import re
from collections import deque

from spec.interfaces import Episode, Move


class _BFSGame:
    """Shared machinery: legal/optimal moves from all_moves + _step; distance by BFS."""

    name = ""

    def _step(self, state, move):  # -> new state or None if illegal
        raise NotImplementedError

    def legal_moves(self, state):
        return [m for m in self.all_moves(state) if self._step(state, m) is not None]

    def apply(self, state, move):
        new = self._step(state, move)
        if new is None:
            raise ValueError(f"illegal move {move.text}")
        return new

    def distance(self, state, cap=12):
        seen, q = {state}, deque([(state, 0)])
        while q:
            s, d = q.popleft()
            if self.solved(s):
                return d
            if d >= cap:
                continue
            for m in self.legal_moves(s):
                n = self._step(s, m)
                if n not in seen:
                    seen.add(n)
                    q.append((n, d + 1))
        return None

    def optimal_moves(self, state):
        d = self.distance(state)
        if not d:
            return []
        return [m for m in self.legal_moves(state) if self.distance(self._step(state, m)) == d - 1]

    def reset(self, seed, level):
        rng = random.Random(f"{self.name}-{seed}-{level}")
        for _ in range(10000):
            s = self._random_state(rng, level)
            if s is not None and self.distance(s) == level:
                return s
        raise RuntimeError("no state at level")


class Hanoi(_BFSGame):
    name = "hanoi"
    N = 3

    def rules(self):
        return ("Tower of Hanoi with 3 disks (1 smallest, 3 largest) on pegs A, B, C. A move takes the top disk of one peg "
                "and puts it on another peg. A disk may never be placed on a smaller disk. Goal: all disks on peg C. "
                "Write a move as source->target, for example A->C.")

    def hint(self):
        return ("Strategy: the top disk of a peg is the last number listed. Only move from a peg that is not empty. "
                "Check that the target peg is empty or its top disk is larger than the disk you move. "
                "To free a large disk, first move the smaller disks to the spare peg.")

    def render(self, state):
        return " | ".join(f"Peg {p}: {' '.join(map(str, d)) if d else 'empty'}" for p, d in zip("ABC", state)) + \
            " (disks listed bottom to top)"

    def all_moves(self, state):
        return [Move(f"{a}->{b}") for a in "ABC" for b in "ABC" if a != b]

    def _step(self, state, move):
        a, b = "ABC".index(move.text[0]), "ABC".index(move.text[-1])
        if not state[a] or (state[b] and state[b][-1] < state[a][-1]):
            return None
        pegs = [list(p) for p in state]
        pegs[b].append(pegs[a].pop())
        return tuple(tuple(p) for p in pegs)

    def why_illegal(self, state, move):
        a, b = "ABC".index(move.text[0]), "ABC".index(move.text[-1])
        if not state[a]:
            return f"peg {move.text[0]} is empty"
        return f"disk {state[a][-1]} cannot go on the smaller disk {state[b][-1]}"

    def solved(self, state):
        return len(state[2]) == self.N

    def parse(self, text):
        m = re.search(r"\b([ABC])\s*(?:->|→|to|-)\s*([ABC])\b", text)
        return Move(f"{m.group(1)}->{m.group(2)}") if m and m.group(1) != m.group(2) else None

    def _random_state(self, rng, level):
        pegs = [[], [], []]
        for d in range(self.N, 0, -1):
            pegs[rng.randrange(3)].append(d)
        return tuple(tuple(p) for p in pegs)


class Countdown(_BFSGame):
    name = "countdown"
    LO, HI = 0, 60

    def rules(self):
        return ("Number game. You have a current number and a target number. Each move applies one of the allowed "
                f"operations to the current number. The number must stay between {self.LO} and {self.HI}. "
                "Goal: make the current number equal to the target. Write a move as the operation itself, for example +3.")

    def hint(self):
        return ("Strategy: compare current and target. If the target is far above, doubling helps; if slightly above, add; "
                "if below, subtract. Compute the result of your move before you answer and check it stays in range.")

    def render(self, state):
        v, t, ops = state
        return f"Current: {v} | Target: {t} | Allowed operations: {', '.join(ops)}"

    def all_moves(self, state):
        return [Move(o) for o in state[2]]

    def _step(self, state, move):
        v, t, ops = state
        if move.text not in ops:
            return None
        k = int(move.text[1:])
        n = {"+": v + k, "-": v - k, "*": v * k}[move.text[0]]
        return (n, t, ops) if self.LO <= n <= self.HI else None

    def why_illegal(self, state, move):
        if move.text not in state[2]:
            return f"{move.text} is not an allowed operation"
        return f"the result would leave the range {self.LO} to {self.HI}"

    def solved(self, state):
        return state[0] == state[1]

    def parse(self, text):
        m = re.search(r"([+\-*x×])\s*(\d+)", text)
        if not m:
            return None
        return Move(("*" if m.group(1) in "x×" else m.group(1)) + m.group(2))

    def _random_state(self, rng, level):
        ops = (f"+{rng.randint(2, 5)}", f"-{rng.randint(1, 4)}", "*2")
        return (rng.randint(1, 20), rng.randint(1, 40), ops)


class Grid(_BFSGame):
    name = "grid"
    SIZE = 4
    DIRS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

    def rules(self):
        return ("Grid walk on a 4x4 grid. Rows are numbered 1 to 4 from top to bottom, columns 1 to 4 from left to right. "
                "Each move goes one cell up, down, left or right. up decreases the row, down increases the row, left "
                "decreases the column, right increases the column. You cannot leave the grid or enter a wall. "
                "Goal: reach the goal cell. Write a move as one word: up, down, left or right.")

    def hint(self):
        return ("Strategy: compare your row with the goal row and your column with the goal column. Move in a direction "
                "that reduces a difference. Before you answer, compute the cell you would enter and check it is inside "
                "the grid and not a wall; if it is blocked, go around.")

    def render(self, state):
        pos, goal, walls = state
        w = ", ".join(f"(row {r}, col {c})" for r, c in walls)
        return f"You: (row {pos[0]}, col {pos[1]}) | Goal: (row {goal[0]}, col {goal[1]}) | Walls: {w}"

    def all_moves(self, state):
        return [Move(d) for d in self.DIRS]

    def _step(self, state, move):
        pos, goal, walls = state
        dr, dc = self.DIRS[move.text]
        n = (pos[0] + dr, pos[1] + dc)
        if not (1 <= n[0] <= self.SIZE and 1 <= n[1] <= self.SIZE) or n in walls:
            return None
        return (n, goal, walls)

    def why_illegal(self, state, move):
        pos, goal, walls = state
        dr, dc = self.DIRS[move.text]
        n = (pos[0] + dr, pos[1] + dc)
        return f"(row {n[0]}, col {n[1]}) is a wall" if n in walls else "that would leave the grid"

    def solved(self, state):
        return state[0] == state[1]

    def parse(self, text):
        m = re.search(r"\b(up|down|left|right)\b", text.lower())
        return Move(m.group(1)) if m else None

    def _random_state(self, rng, level):
        cells = [(r, c) for r in range(1, 5) for c in range(1, 5)]
        pick = rng.sample(cells, 5)
        return (pick[0], pick[1], tuple(sorted(pick[2:])))


class ExactVerifier:
    """Independent of the runner: recomputes legality from the rules and replays whole episodes."""

    def legal(self, game, state, move):
        return move in game.all_moves(state) and move in game.legal_moves(state)

    def replay(self, game, episode: Episode):
        state = game.reset(episode.seed, episode.level)
        for text in episode.moves:
            move = Move(text)
            if not self.legal(game, state, move):
                return False
            state = game.apply(state, move)
        return game.solved(state) == episode.solved


GAMES = {g.name: g for g in (Hanoi(), Countdown(), Grid())}
