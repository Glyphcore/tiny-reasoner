"""Source of truth: types only (Protocols and dataclasses). No behaviour lives here.

Every class below has a box in spec/diagrams.md; tools/spec_check.py fails on drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Move:
    text: str  # canonical form, e.g. "A->C", "+3", "up"


class Game(Protocol):
    name: str

    def reset(self, seed: int, level: int) -> Any: ...
    def rules(self) -> str: ...
    def hint(self) -> str: ...
    def render(self, state: Any) -> str: ...
    def all_moves(self, state: Any) -> list[Move]: ...
    def legal_moves(self, state: Any) -> list[Move]: ...
    def optimal_moves(self, state: Any) -> list[Move]: ...
    def why_illegal(self, state: Any, move: Move) -> str: ...
    def apply(self, state: Any, move: Move) -> Any: ...
    def solved(self, state: Any) -> bool: ...
    def parse(self, text: str) -> Move | None: ...


class Verifier(Protocol):
    def legal(self, game: Game, state: Any, move: Move) -> bool: ...
    def replay(self, game: Game, episode: Episode) -> bool: ...


@dataclass
class Trace:
    step: int
    text: str  # raw model reply: reasoning line plus move line
    tokens: int  # generated tokens (trace length)
    move: str | None  # parsed canonical move, None if unreadable
    legal: bool
    optimal: bool
    after_pushback: bool


@dataclass
class Episode:
    game: str
    seed: int
    level: int  # optimal solution length from the start state
    traces: list[Trace]
    moves: list[str]  # accepted (legal) moves in order
    solved: bool
    verified: bool  # Verifier.replay agreed with the runner


class Policy(Protocol):
    model_id: str
    revision: str

    def reply(self, messages: list[dict[str, str]], max_new_tokens: int) -> tuple[str, int]: ...
    def nll(self, text: str) -> tuple[float, int]: ...


@dataclass
class Eval:
    split: str  # "dev" or "heldout"
    seeds: dict[str, list[int]]  # game -> seeds
    levels: list[int]
    max_new_tokens: int
    chat_max_new_tokens: int
    probe_states: int  # teacher-forced move probes per game
    solve: float = 0.0
    legal: float = 0.0
    optimal: float = 0.0
    recover: float = 0.0
    steps: float = 0.0
    trace_len: float = 0.0
    probe_legal: float = 0.0
    probe_opt: float = 0.0
    ppl: float = 0.0
    chats_ok: str = ""  # "k/n"


@dataclass
class WeightEdit:
    id: str
    family: str  # steer_bias | ortho | scale | lowrank | task_vector
    description: str
    ops: list[dict[str, Any]]  # recipe, reproducible from config.json
    tensors: list[str] = field(default_factory=list)
    before: dict[str, str] = field(default_factory=dict)  # tensor -> sha256 (or "absent")
    after: dict[str, str] = field(default_factory=dict)


@dataclass
class Adapter:
    name: str
    rank: int
    alpha: float  # delta = (alpha / rank) * B @ A
    targets: dict[str, tuple[Any, Any]]  # tensor name -> (A [r, in], B [out, r])


@dataclass
class Run:
    id: str
    model: str
    revision: str
    license: str
    device: str
    dtype: str
    threads: int
    prompt: str  # "plain" or "hint" (prompt-only baseline)
    edit: WeightEdit | None
    eval: Eval
    base_hash: str  # hash of all weights before the edit
    reverted_hash: str  # hash of all weights after revert; must equal base_hash
    seconds: float
