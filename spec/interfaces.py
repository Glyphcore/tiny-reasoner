"""tiny-reasoner spec: Protocols and dataclasses only (types, no logic).

Source of truth. `spec/diagrams.md` has one mermaid class box per type here; `tools/spec_check.py` fails on drift
between this file, the diagram boxes and the implementations in `tr/`.

Conventions
- An implementation subclasses the Protocol explicitly (`class Hanoi(Game):`) so spec_check can find it.
- `State` and `Action` are game-specific, hashable, JSON-serialisable via `Game.encode/decode`.
- Hashes are sha256 hex of the tensor bytes in fp16 row-major order (`tr.edits.hashing.tensor_hash`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, Sequence

JSON = Mapping[str, Any]
State = Any
Action = Any
Tensor = Any  # torch.Tensor


# ---------------------------------------------------------------- games

@dataclass(frozen=True)
class Move:
    """One move the model proposed: raw text plus the parsed action (None when unparseable)."""
    text: str
    action: Optional[Action]


class Game(Protocol):
    """A puzzle with discrete moves, difficulty levels and exact rules. Pure functions over State."""
    name: str
    levels: Sequence[int]  # valid difficulty levels, ascending

    def new(self, level: int, seed: int) -> State: ...
    def legal_moves(self, state: State) -> Sequence[Action]: ...
    def apply(self, state: State, action: Action) -> State: ...
    def optimal_steps(self, state: State) -> int: ...
    def render(self, state: State) -> str: ...        # board as text for the prompt
    def rules(self) -> str: ...                        # rules + move format, for the system prompt
    def parse(self, text: str) -> Optional[Action]: ...  # model text -> action, None if malformed
    def format(self, action: Action) -> str: ...       # canonical move text
    def encode(self, state: State) -> JSON: ...
    def decode(self, data: JSON) -> State: ...


class Verifier(Protocol):
    """Exact checker for one Game. No model judgement anywhere."""
    game: Game

    def legal(self, state: State, action: Action) -> bool: ...
    def solved(self, state: State) -> bool: ...
    def explain(self, state: State, action: Action) -> str: ...  # why a move is illegal, for the chat pushback


# ---------------------------------------------------------------- model

@dataclass(frozen=True)
class Message:
    role: str  # system | user | assistant
    content: str


@dataclass
class Generation:
    text: str                 # final answer text (thinking stripped)
    reasoning: str            # thinking text ("" in non-thinking mode)
    n_tokens: int             # new tokens generated
    truncated: bool           # hit max_new_tokens


@dataclass(frozen=True)
class PolicyConfig:
    model: str                 # HF id
    revision: str              # commit hash
    variant: str               # "base" | edit id | adapter id
    dtype: str = "float16"
    device: str = "cuda"
    thinking: bool = True
    max_new_tokens: int = 512
    temperature: float = 0.0   # 0 = greedy
    seed: int = 0


class Policy(Protocol):
    """Model wrapper. Owns the model and tokenizer; edits and adapters mutate `model` in place."""
    config: PolicyConfig
    model: Any
    tokenizer: Any

    def generate(self, messages: Sequence[Message]) -> Generation: ...
    def nll(self, text: str) -> tuple[float, int]: ...     # (sum negative log-likelihood, n tokens) for perplexity
    def tensor(self, name: str) -> Tensor: ...             # live parameter by name
    def tensor_names(self) -> Sequence[str]: ...


# ---------------------------------------------------------------- episodes and traces

@dataclass
class Step:
    turn: int
    prompt: str                # user message for this step
    reasoning: str
    move: Optional[Move]
    legal: bool
    solved_after: bool
    n_tokens: int
    error: Optional[str]       # "unparseable" | "illegal" | "truncated" | None


@dataclass
class Trace:
    """Step-by-step reasoning record of one episode."""
    game: str
    level: int
    seed: int
    mode: str                  # "solo" | "chat"
    variant: str
    steps: list[Step] = field(default_factory=list)
    solved: bool = False
    optimal_steps: int = 0
    n_illegal: int = 0
    n_unparseable: int = 0
    reasoning_tokens: int = 0
    wall_s: float = 0.0


@dataclass(frozen=True)
class EpisodeConfig:
    mode: str = "solo"         # solo: one move per turn, board re-rendered | chat: user pushes back on illegal moves
    max_turns: int = 40
    max_illegal: int = 3       # episode fails after this many illegal/unparseable moves in a row
    show_legal: bool = False   # list legal moves in the prompt (prompt-only baseline knob)
    cot_prompt: bool = False   # "think step by step" in the user turn (prompt-only baseline knob)


class Episode(Protocol):
    """Plays one game instance with one policy to a verified end, producing a Trace."""
    game: Game
    verifier: Verifier
    policy: Policy
    config: EpisodeConfig

    def play(self, level: int, seed: int) -> Trace: ...
    def messages(self) -> Sequence[Message]: ...  # transcript of the last play


# ---------------------------------------------------------------- evaluation

@dataclass(frozen=True)
class GameSuite:
    game: str
    levels: Sequence[int]
    seeds: Sequence[int]
    episode: EpisodeConfig = EpisodeConfig()


@dataclass(frozen=True)
class TextSuite:
    path: str                  # held-out text, one document per line
    max_docs: int = 200


@dataclass(frozen=True)
class ChatSuite:
    path: str                  # JSON: list of {id, turns:[user strings]}; assistant replies are generated
    max_chats: int = 20


@dataclass
class GameMetrics:
    game: str
    level: int
    n: int
    solve_rate: float
    legal_rate: float          # legal moves / all moves
    mean_steps: float          # moves in solved episodes
    mean_optimal: float
    mean_reasoning_tokens: float
    mean_wall_s: float


@dataclass
class TextMetrics:
    path: str
    n_docs: int
    n_tokens: int
    ppl: float


@dataclass
class ChatMetrics:
    path: str
    n_chats: int
    mean_reply_tokens: float
    truncated_rate: float
    repeat_rate: float         # replies whose last 20 tokens repeat an earlier 20-gram
    transcripts_dir: str       # saved side-by-side material


@dataclass
class EvalResult:
    variant: str
    games: list[GameMetrics] = field(default_factory=list)
    text: Optional[TextMetrics] = None
    chats: Optional[ChatMetrics] = None
    game_chats: list[GameMetrics] = field(default_factory=list)  # games played in chat mode


class Eval(Protocol):
    """Runs suites against one policy, writes `runs/<id>/metrics.jsonl` and one summary line."""
    run: "Run"

    def games(self, policy: Policy, suite: GameSuite) -> list[GameMetrics]: ...
    def text(self, policy: Policy, suite: TextSuite) -> TextMetrics: ...
    def chats(self, policy: Policy, suite: ChatSuite) -> ChatMetrics: ...
    def all(self, policy: Policy, games: Sequence[GameSuite], text: TextSuite, chats: ChatSuite) -> EvalResult: ...


# ---------------------------------------------------------------- runs

@dataclass(frozen=True)
class RunConfig:
    policy: PolicyConfig
    games: Sequence[GameSuite]
    text: TextSuite
    chats: ChatSuite
    edit: Optional[str] = None      # WeightEdit id applied for this run
    adapter: Optional[str] = None   # Adapter id applied for this run
    note: str = ""


class Run(Protocol):
    """One reproducible evaluation: `runs/<id>/{config.json, metrics.jsonl, summary.txt, chats/, traces/}`."""
    id: str
    dir: str
    config: RunConfig

    def write_config(self) -> None: ...
    def log(self, record: JSON) -> None: ...              # append one line to metrics.jsonl
    def summary(self, result: EvalResult) -> str: ...      # one line, also written to summary.txt
    def save_trace(self, trace: Trace) -> str: ...
    def save_chat(self, chat_id: str, messages: Sequence[Message]) -> str: ...


# ---------------------------------------------------------------- weight changes

@dataclass
class TensorChange:
    name: str
    hash_before: str
    hash_after: str
    how: JSON                  # e.g. {"op": "add_rank1", "scale": 0.5} or {"op": "scale_rows", "rows": [...]}


@dataclass(frozen=True)
class LoraTarget:
    tensor: str                # weight name, e.g. model.layers.12.mlp.down_proj.weight
    A: Tensor                  # (r, in)
    B: Tensor                  # (out, r)
    alpha: float = 1.0         # delta = alpha * B @ A


class Adapter(Protocol):
    """Hand-built low-rank delta in LoRA form. A and B are written from analysis, never learned."""
    id: str
    rank: int
    targets: Sequence[LoraTarget]

    def merge(self, policy: Policy) -> "WeightEdit": ...    # fold into weights, returns the edit with hashes
    def save(self, path: str) -> None: ...                  # safetensors + json meta
    def load(self, path: str) -> "Adapter": ...


class WeightEdit(Protocol):
    """A direct tensor change with before/after hashes. `revert` must restore the before-hashes exactly."""
    id: str
    kind: str                  # steer_bias | orth_project | rank1 | head_scale | neuron_ablate | task_vector | lora_merge
    changes: list[TensorChange]
    applied: bool

    def apply(self, policy: Policy) -> None: ...
    def revert(self, policy: Policy) -> None: ...
    def verify(self, policy: Policy, expect: str) -> bool: ...  # expect "before" | "after": live hashes match
    def save(self, path: str) -> None: ...                      # json meta + safetensors of the deltas
    def load(self, path: str) -> "WeightEdit": ...
